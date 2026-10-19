// Independent old/new byte regression; inputs are finite, outputs may include NaNs and infinities.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <type_traits>
#include "quant/mlx.hpp"
#include "quant/mlx_tiles.hpp"
using namespace tf::rocm;
static void ck(hipError_t e) { if(e!=hipSuccess) { fprintf(stderr,"HIP: %s\n",hipGetErrorString(e)); exit(1); } }
struct Mem { void* p; explicit Mem(size_t n) {ck(hipMalloc(&p,n));} ~Mem(){hipFree(p);} };
template<class U> static void upload(Mem& m,const std::vector<U>& x){ck(hipMemcpy(m.p,x.data(),x.size()*sizeof(U),hipMemcpyHostToDevice));}
static uint32_t rng=0x31415926;
static uint32_t next(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
template<class T> static uint16_t bits(float f) {
 if constexpr(std::is_same_v<T,DotF16>) { __half h=__float2half(f);uint16_t v;memcpy(&v,&h,2);return v; }
 else { hip_bfloat16 h(f);uint16_t v;memcpy(&v,&h,2);return v; }
}
// Mode 0 plain fp32,1 activation out,2 relu2,3 swiglu,4 routed,5 two sides.
template<class T,int B,int R,int CB> static void one(const char* out,int group,int pattern,int mode) {
 const int k=group*3,n=37,rows=R+1,groups=k/group,experts=mode==4?2:1;
 const int lpc_log2=2,lpc=4,per=4*(32/lpc)*(mode==3?CB/2:CB);
 const int width=mode==3?2*n:n, words=width*k*B/32;
 std::vector<uint16_t> x(rows*k),sc(experts*width*groups),bi(sc.size());
 std::vector<uint32_t> w(experts*words,0);
 for(size_t i=0;i<x.size();++i) {
   float f=static_cast<int>(next()%65)-32; f*=0.03125f;
   x[i]=bits<T>(f);
   if(pattern==1) x[i]=(i&1)?0x8000:0;
   if(pattern==2) x[i]=std::is_same_v<T,DotF16>?0x7bff:0x7f7f;
   if(pattern==3) x[i]=bits<T>((i%k<k/2?1.f:-1.f)*(1.f+float(i%8)));
   if(pattern==4) x[i]=uint16_t((i&1?0x8000:0)|(1+(next()%127)));
 }
 for(size_t i=0;i<sc.size();++i) {
   sc[i]=bits<T>((static_cast<int>(next()%15)-7)*0.125f);
   bi[i]=bits<T>((static_cast<int>(next()%15)-7)*0.0625f);
   if(pattern==2) { sc[i]=0x7f7f; if(i%groups==1)sc[i]|=0x8000;bi[i]=bits<T>(0.f); }
 }
 for(int e=0;e<experts;++e)for(int c=0;c<width;++c)for(int j=0;j<k;++j){
   uint32_t q=next()&((1u<<B)-1);if(pattern==2)q=(1u<<B)-1;
   size_t bit=(static_cast<size_t>(e)*width*k+c*k+j)*B;
   w[bit/32]|=q<<(bit%32);if(bit%32+B>32)w[bit/32+1]|=q>>(32-bit%32);
 }
 Mem dx(x.size()*2),ds(sc.size()*2),db(bi.size()*2),dw(w.size()*4);
 upload(dx,x);upload(ds,sc);upload(db,bi);upload(dw,w);
 const size_t count=static_cast<size_t>(rows)*n;
 const bool half=mode!=0 && mode!=4;
 Mem dy(count*(half?2:4)),dy2(count*(half?2:4));
 ck(hipMemset(dy.p,0xa5,count*(half?2:4)));ck(hipMemset(dy2.p,0xa5,count*(half?2:4)));
 Affine a{};a.x=dx.p;a.words=(uint32_t*)dw.p;a.scale={ds.p,pattern==2?1:(std::is_same_v<T,DotF16>?2:1)};a.bias={db.p,a.scale.kind};
 a.m=rows;a.n=width;a.k=k;a.bits=B;a.group=group;a.fp16=std::is_same_v<T,DotF16>;
 if(half)a.out16=(__half*)dy.p;else a.out=(float*)dy.p;
 StreamSides sides{};int items=1;
 std::vector<int> plan={1,0,rows},members(rows);for(int i=0;i<rows;++i)members[i]=rows-1-i;
 Mem dp(plan.size()*4),dm(members.size()*4);upload(dp,plan);upload(dm,members);
 if(mode==4){a.route.items=(int*)dp.p;a.route.members=(int*)dm.p;a.route.x_div=1;}
 int blocks=(n+per-1)/per;
 if(mode==3){sides.pair_cols=n;sides.limit=0;}
 if(mode==5){sides.count=2;sides.first[0]=0;sides.first[1]=blocks;sides.first[2]=blocks*2;sides.out_half=half;
   for(int s=0;s<2;++s){sides.words[s]=a.words;sides.scale[s]=a.scale.p;sides.bias[s]=a.bias.p;sides.out[s]=s?dy2.p:dy.p;sides.n[s]=n;}blocks*=2;}
 dim3 grid(blocks,(rows+R-1)/R,items),threads(128);int shift=group==32?0:group==64?1:2;
 if(mode==2)hipLaunchKernelGGL((affine_dot2_stream_relu2<T,B,R,CB>),grid,threads,0,0,a,sides,lpc_log2,shift);
 else if(mode==3)hipLaunchKernelGGL((affine_dot2_stream<T,B,R,CB,true>),grid,threads,0,0,a,sides,lpc_log2,shift);
 else hipLaunchKernelGGL((affine_dot2_stream<T,B,R,CB,false>),grid,threads,0,0,a,sides,lpc_log2,shift);
 ck(hipGetLastError());ck(hipDeviceSynchronize());
 std::vector<unsigned char> bytes(count*(half?2:4));ck(hipMemcpy(bytes.data(),dy.p,bytes.size(),hipMemcpyDeviceToHost));
 std::string name=std::string(out)+"/"+(std::is_same_v<T,DotF16>?"f16":"bf16")+"_b"+std::to_string(B)+"_r"+std::to_string(R)+"_cb"+std::to_string(CB)+"_g"+std::to_string(group)+"_p"+std::to_string(pattern)+"_m"+std::to_string(mode)+".bin";
 FILE*f=fopen(name.c_str(),"wb");if(!f){perror("fopen");exit(1);}fwrite(bytes.data(),1,bytes.size(),f);
 if(mode==5){ck(hipMemcpy(bytes.data(),dy2.p,bytes.size(),hipMemcpyDeviceToHost));fwrite(bytes.data(),1,bytes.size(),f);}fclose(f);
}
template<class T,int B,int R,int CB>static void tile(const char* out){for(int g:{32,64,128})for(int p=0;p<5;++p)for(int m=0;m<6;++m)one<T,B,R,CB>(out,g,p,m);}
template<class T>static void suite(const char* out){tile<T,2,1,2>(out);tile<T,3,2,4>(out);tile<T,4,1,8>(out);tile<T,5,4,2>(out);tile<T,6,4,4>(out);tile<T,8,2,8>(out);}
int main(int argc,char**argv){if(argc!=2){fprintf(stderr,"usage: finite OUTPUT_DIR\n");return 2;}suite<DotF16>(argv[1]);
#if TF_DOT2_BF16
 suite<DotBF16>(argv[1]);
#endif
 puts("FINITE_DUMP_DONE");}

#include <cstdint>
#include <cstdio>
#include <cstdlib>
using U=uint32_t;
bool finite(U a){return(a&0x7f800000u)!=0x7f800000u;}bool nz(U a){return(a&0x7fffffffu)!=0;}
bool product(U a,U b){U ae=a>>23&255,be=b>>23&255,s=ae+be;return(finite(a)&finite(b)&nz(a)&nz(b))&((ae==0)|(be==0)|(s<=129)|(s>=380));}
bool product_fast(U a,U b){U ae=a>>23&255,be=b>>23&255;if(ae>0&&ae<255&&be>0&&be<255&&ae+be>129&&ae+be<380)return false;return product(a,b);}
bool divrisk(U a,U b){int ae=a>>23&255,be=b>>23&255;return(finite(a)&finite(b)&nz(a)&nz(b))&((ae==0)|(be==0)|(ae-be<=-125)|(ae-be>=126));}
bool divfast(U a,U b){int ae=a>>23&255,be=b>>23&255;if(ae>0&&ae<255&&be>0&&be<255&&ae-be> -125&&ae-be<126)return false;return divrisk(a,b);}
bool sum(U a,U b,U c,U d,U r){U e=r>>23&255;return(finite(a)&finite(b)&finite(c)&finite(d))&((e>=253)|((e==0)&((nz(a)&nz(b))|(nz(c)&nz(d)))));}
bool sumfast(U a,U b,U c,U d,U r){U e=r>>23&255;if(e>0&&e<253)return false;return sum(a,b,c,d,r);}
bool exprisk(U gap,U v){return finite(gap)&&((gap>>31)!=0)&&nz(gap)&&((v&0x7f800000u)==0);}
bool expfast(U gap,U v){if((v&0x7f800000u)!=0)return false;return exprisk(gap,v);}
int main(){uint64_t pairs=0,sums=0,random=0;U frac[]={0,1,0x400000,0x7fffff};
 for(U ae=0;ae<256;++ae)for(U be=0;be<256;++be)for(U sa=0;sa<2;++sa)for(U sb=0;sb<2;++sb)for(U af:frac)for(U bf:frac){U a=sa<<31|ae<<23|af,b=sb<<31|be<<23|bf;if(product(a,b)!=product_fast(a,b)||divrisk(a,b)!=divfast(a,b)||exprisk(a,b)!=expfast(a,b))abort();++pairs;}
 U cls[]={0,0x80000000,1,0x80000001,0x3f800000,0xbf800000,0x7f800000,0xff800000,0x7fc00000,0xffc00000};
 for(U a:cls)for(U b:cls)for(U c:cls)for(U d:cls)for(U e=0;e<256;++e)for(U sign=0;sign<2;++sign)for(U f:frac){U r=sign<<31|e<<23|f;if(sum(a,b,c,d,r)!=sumfast(a,b,c,d,r))abort();++sums;}
 U seed=1502;auto next=[&](){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;};
 for(U i=0;i<1000000;++i){U a=next(),b=next(),c=next(),d=next(),r=next();if(sum(a,b,c,d,r)!=sumfast(a,b,c,d,r)||product(a,b)!=product_fast(a,b)||divrisk(a,b)!=divfast(a,b)||exprisk(a,b)!=expfast(a,b))abort();++random;}
 for(U initial=0;initial<2;++initial)for(U bits=0;bits<256;++bits){U old=initial,fast=initial;for(U j=0;j<8;++j){old|=(bits>>j)&1;if(fast==0)fast=(bits>>j)&1;}if(old!=fast)abort();}
 printf("{\"pair_classes\":%llu,\"sum_classes\":%llu,\"random_words\":%llu,\"sticky_sequences\":512,\"mismatches\":0}\n",pairs,sums,random);
}

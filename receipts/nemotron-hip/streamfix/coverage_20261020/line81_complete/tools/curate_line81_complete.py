from pathlib import Path
import json,hashlib,re,argparse
p=argparse.ArgumentParser();p.add_argument('receipt_root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();H=a.receipt_root;R=H/'coverage_20261020';D=R/'line81_recovery_20261021';OLD=R/'continuation_all4_public';A=H/'optimization/matrix729_line81_analysis';assert not a.output.exists();a.output.mkdir(parents=True);sha=lambda b:hashlib.sha256(b).hexdigest();rows=[];excluded=[];oldmanifest=json.loads((OLD/'public_manifest.json').read_text());oldmap={r['path']:r for r in oldmanifest['files']};homes=set()
for f in(D/'raw').rglob('*'):
 if f.is_file()and f.suffix!='.pyc':homes.update(re.findall(rb'/home/[A-Za-z0-9_-]+',f.read_bytes()))
def scrub(b):
 s=b.decode().replace(str(H),'<RECEIPT_ROOT>')
 for home in homes:s=s.replace(home.decode(),'<REMOTE_HOME>')
 return s.encode()
def write(src,dest,origin,prior=None):
 raw=src.read_bytes()
 if src.suffix=='.pyc':excluded.append({'path':dest,'original_bytes':len(raw),'original_sha256':sha(raw),'reason':'DerivedPythoncache omitted; source retained and exactoriginal localarchive retained'});return
 public=scrub(raw);f=a.output/dest;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(public)
 if re.search(r'\.c[18]\.json$',src.name)or src.name.endswith('.ids.json'):assert raw==public
 entry={'path':dest,'origin':origin,'original_bytes':len(raw),'original_sha256':sha(raw),'public_bytes':len(public),'public_sha256':sha(public),'redacted':raw!=public}
 if prior:entry.update(original_bytes=prior['original_bytes'],original_sha256=prior['original_sha256'],original_to_previously_published_public_sha256=prior['public_sha256'],redacted=prior['redacted']or raw!=public)
 assert not re.search((b'/'+b'home/[A-Za-z0-9_-]+|/'+b'Users/|/'+b'private/tmp/|par1-'+b'cs25'),public),dest
 rows.append(entry)
for f in sorted((OLD/'raw').rglob('*')):
 if f.is_file():n=str(f.relative_to(OLD));write(f,'opening_pairs/'+n,'Previouslypublishedsourcebound3openingO_Npairs_andfailedsupervisor',oldmap[n])
for f in sorted((D/'raw').rglob('*')):
 if f.is_file():write(f,'controls/'+str(f.relative_to(D/'raw')),'Actualrecoveredline81controls_original')
for n in ['download_manifest.json','remote_inventory.json','recovery_integrity_audit.json','independent_recovery_audit.json','independent_alternation_reduction.json','independent_control_rates.json']:write(D/n,'audits/'+n,'Actualpostrunrecovery_orindependentaudit')
for n in ['independent_von_final_audit.json','COLLECTION_RELEASE.json']:write(R/'continuation_von'/n,'audits/opening_V_ON/'+n,'Actualopeningpair_collection_release')
write(A/'reduce_alternation.py','analysis/reduce_alternation.py','Postrunreviewedofflineanalysis607bd252')
write(Path(__file__),'tools/curate_line81_complete.py','Postruncurationtool')
selection=json.loads((A/'selection.actual_complete.json').read_text())
for b in selection['blocks']:
 for e in b['entries']:
  original=e['run_root'];e['original_state_sha256']=e['state_sha256']
  if 'line81_recovery_20261021/raw/'in original:e['run_root']='controls/'+original.split('line81_recovery_20261021/raw/',1)[1]
  else:e['run_root']='opening_pairs/raw/'+original.split('/raw/',1)[1]
  e['state_sha256']=sha((a.output/e['run_root']/'state.json').read_bytes())
selection['fixtures']={n:{'file':'opening_pairs/raw/matrix729_v_off_pair1/fixtures/'+n+'.ids.json'}for n in ['code_1024_512','prose_16384_1024']};selection['public_binding_scope']='Onlystatepath/hash metadata remapped to privacy-normalizedcopies; all originalbindings preserved, HTTP/SSE/fixturebytes unchanged.'
def generated(n,content):
 f=a.output/n;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(content);rows.append({'path':n,'origin':'Generatedpublicscopeorhashindex','public_bytes':f.stat().st_size,'public_sha256':sha(f.read_bytes())})
generated('analysis/selection.public.json',json.dumps(selection,indent=2)+'\n')
generated('README.md','''# Completed alternating baseline controls

Measuredsource729833615595df66c9f9a438ce9a85061843f010 versus baseline8649e9858b38e9ef3dcd8d579879c6512a363050. This is source-bound OLD/NEW/OLD coverage for all16requested shape/concurrency/mode/rig cases: exact1024input/512outputcode and16384input/1024outputprose;C1/C8;draftsOFF/ON;W7800TP2/V620TP1. All48selectedcells contain864requests including648measured, one warmup plus3measured batches percell. Everycomplete request validateslength/finish/cache0/matchingfullfingerprints foritsfixture. Allrequests/SSE and publicinputarrays remain exact originalbytes. Serverreportedtokenhashes are retained; outputtokenIDarrays were notavailable forindependent tokenhash recomputation.

Sixnewcontrolholds/24cells were actuallycompleted. VON,VOFF,WON appendoneclosingOLD hold to eachpreviouslycompleted source-bound sameboot O/Npair. Thosecontrols have longintervening gaps; no scheduling/thermal/host-pressure cause is inferred. WOFF insteaduses afreshsameboot O/N/O3holdblock: the historical pre-reboot WOFFpair is excluded from thiscontrolcampaign and remains independentlypublished history. The fourcases are notone uninterrupted ABBA experiment or confidenceproof. No automaticdriftthreshold, acceptancewaiver, fullCONTRIBUTINGqualification orreadyaction is inferred.

Actualsupervisorcomplete timestamp2026-10-20T22:52:47Z, measuredboot9064ffa5-1d25-4059-a521-5e5d69d03a36. All4blocklauncher/coordinator/observer statuses0, recordednativecleanupguards passed beforeadvancing/finishing. Rank0/TP1nativeexit0; TP2workerexit-15 is accepted onlyifalivebefore explicitintentionalteardown, exactreason, noforcedkill and ownedgroupgone. Rawstatuses remainunchanged. A laterbootccd3c480-76df-44a2-a67f-ee1db46a46e5 was observed duringrecovery; the completedoldbootmeasurements are not relabelled, and no rebootcause/gracefulOSshutdown is established. The initialsupervisorattempt missinganallocationtemplate failedbeforeanyblock; itsfailurehistory remains intact.

## Metrics and drift

Fullprecisionperrequest/batch/allaggregate metrics are in audits/independent_alternation_reduction.json. Serverprefillelapsed includes schedulerelapsedtime, notpureGPUcompute. Client/serverTTFT, serverdecodeTPS, deliverymeanTPOT and transportchunkgaps are distinct. TruepertokenITL distributions are unavailable becauseSSEchunks maybatchtokens. Aggregatedcompletionend-to-endTPS uses totalcompletiontokens dividedby firstsend-to-lastcompletion batchwalltime; weightedserverprefillserviceTPS and firstcontentwindowproxy haveexplicitotherdenominators. Threeactualbatchvalues, holdmeans/medians and pooledactivebatchwall versuselapsedspanincludinggaps are retained.

For eachmetric the analysisreports OLDfirst/NEW/OLDlast, baselinefirst-to-lastchange, candidatevseachbaseline andcandidatevsthearithmeticbookendmean. The latterisdescriptive, notatemporaldriftmodel orconfidenceinterval. Actualsamebootmonotonicrequestspans andinterveninggaps are explicit; chronology/nonoverlap is checked. Allregressions remainvisible, including WdraftsONcode over5percent and VdraftsONcode over3percent versusdescriptivebookendmeans. Requesteracceptance isnotupstreamacceptance.

## Reproduction and provenance

Run `python3 analysis/reduce_alternation.py . --selection analysis/selection.public.json --output reduction.local.json` offline. The reviewed607bd252 reducer and originalmathematicalvalidations are unchanged. Publicselectionrebindsonly normalizedmetadata statepaths/hashes; originalstateSHAfields andmanifestoriginal/publicmappingsremain. Actualversion-specific native binaries, runtime/librarymaps, checkpointrevision, clients/fixtures/defaultadmission andheldcanonicalGPU/buildlock evidence remain sourcebound. SharedhostCPU/disjointGPUactivity permitted; no physicalhostisolation, fullpinnedmemory or exactphysicalpeak claim.

All357neworiginalfiles remainlocal with the270981120-byte tar SHA256 a088bb6e76978c389ad8cab8648f9c2b05c78b89eadd1c4db5065e7800ac944f. Large tar andderivedPythoncaches are omitted publicly withoriginalSHA/bytes/reason. Publiccopies ofthethreeopeningpairs derivefrom thepreviousimmutable32cellreceipt df70b95290cda79e4550468544a27b6ce70908f9; theiroriginal/publichashes are retained. Historicalinterrupted/orphan/failedsupervisor records are preserved separately andnotpooledintosyntheticquartets. No weights/nativebinaries are included. Onlyexactprivatehome/receiptroot metadata is normalized; requests/SSE/fixturearrays are unchanged.

Theprivate826a/a44runnercorrection isnotthismeasuredsource. Itsqualificationandapplicable standardreference/precision/platform/CI/maintainergates remainseparate. This data package doesnotmarkanyPRready.
''')
assert all(r['public_bytes']<100*1024**2 for r in rows)
m={'schema_version':1,'scope':'Actualcomplete48cellO_N_Odescriptivecontrols; noABBA/confidence/readiness','source_commit':'729833615595df66c9f9a438ce9a85061843f010','baseline_commit':'8649e9858b38e9ef3dcd8d579879c6512a363050','new_original_files':357,'new_archive_bytes':270981120,'new_archive_sha256':'a088bb6e76978c389ad8cab8648f9c2b05c78b89eadd1c4db5065e7800ac944f','new_archive_included':False,'selected_cells':48,'selected_requests':864,'selected_measured_requests':648,'comparison_cases':16,'all_HTTP_SSE_and_fixture_bytes_original':True,'files':rows,'excluded_paths':excluded,'manifest_self_excluded':True};(a.output/'public_manifest.json').write_text(json.dumps(m,indent=2)+'\n');print('files',len(rows),'excluded',len(excluded),'manifestSHA',sha((a.output/'public_manifest.json').read_bytes()))

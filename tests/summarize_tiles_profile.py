"""Summarize nonoverlapping wall-time phases and UI timer/draw observations."""
import argparse
import json
from pathlib import Path
import statistics


def stats(values):
    values = list(values)
    return dict(mean=statistics.mean(values) if values else 0.,
                p95=sorted(values)[int(.95*(len(values)-1))] if values else 0.,
                maximum=max(values,default=0.))


def summarize(path):
    report=json.loads(path.read_text())
    assert report['status']=='passed', path
    result={key:report[key] for key in ('elapsed','object_count','result_tiles','data_counts')}
    for key in ('initial_objects','geometry_sha256','world_geometry_mm_sha256'):
        result[key]=report.get(key)
    memory_path=path.with_name('memory.json')
    if memory_path.exists():
        memory=json.loads(memory_path.read_text())
        result['peak_rss_bytes']=max((row['rss'] for row in memory),default=0)
    result['arguments']=report['args']
    result['windows']=[]
    count=report['args']['count']
    intervals=[(row['tile'],right['seconds']-row['seconds'])
               for row,right in zip(report['rows'],report['rows'][1:])]
    for begin,end in dict.fromkeys(((1,min(100,count)),(max(1,count-99),count),
                                   *((i+1,min(i+500,count)) for i in range(0,count,500)))):
        rows=[row for row in report['rows'] if begin<=row['tile']<=end]
        ticks=[tick for tick in report['ticks'] if tick['published'] and begin<=tick['tiles']<=end]
        window=dict(begin=begin,end=end)
        for name in ('tile_seconds','gltf_create','explicit_update','finalize','rollback_snapshots',
                     'prepare','preserve_enter','preserve_exit'):
            window[name]=stats(row.get(name,0.) for row in rows)
        window['checkpoint']=stats(tick.get('checkpoint_seconds',tick['duration']-
            next(row['tile_seconds'] for row in rows if row['tile']==tick['tiles'])) for tick in ticks)
        window['tick']=stats(tick['duration'] for tick in ticks)
        window['heartbeat']=stats(gap for tile,gap in report['heartbeat'] if begin<=tile<=end)
        window['draw']=stats(gap for tile,gap in report['draw_intervals'] if begin<=tile<=end)
        window['publication_interval']=stats(gap for tile,gap in intervals if begin<=tile<=end)
        result['windows'].append(window)
    result['metrics']=report['metrics']
    result['idle_heartbeat']=report['idle_heartbeat']
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    report={path.parent.name:summarize(path) for path in args.root.glob('*/profile.json')}
    (args.root/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for name,result in report.items():
        first,last=result['windows'][0],result['windows'][min(1,len(result['windows'])-1)]
        print(name,'seconds',round(result['elapsed'],3),'objects',result['object_count'])
        for label,window in (('first',first),('last',last)):
            print(' ',label,window['begin'],window['end'],
                  ', '.join(f'{key} {1000*window[key]["mean"]:.2f}ms'
                            for key in ('tick','tile_seconds','explicit_update','finalize','rollback_snapshots','checkpoint','heartbeat')))

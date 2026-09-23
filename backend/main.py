from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
import os, sqlite3, hashlib, uuid, math, time, json
import requests
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor

BASE=os.path.dirname(os.path.abspath(__file__))
DB=os.path.join(BASE,'ner_logistics.db')
UPLOADS=os.path.join(BASE,'uploads')
os.makedirs(UPLOADS,exist_ok=True)

app=FastAPI(title='NER Logistics Intelligence API',version='2.0')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_credentials=True,allow_methods=['*'],allow_headers=['*'])
app.mount('/uploads',StaticFiles(directory=UPLOADS),name='uploads')

# ---------------- Database ----------------
def db():
    c=sqlite3.connect(DB); c.row_factory=sqlite3.Row; return c

def init_db():
    c=db(); cur=c.cursor()
    cur.executescript('''
    CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT NOT NULL,email TEXT UNIQUE NOT NULL,password_hash TEXT NOT NULL,created_at TEXT DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS route_analyses(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER,source TEXT,destination TEXT,vehicle_type TEXT,cargo_type TEXT,cargo_weight REAL,priority TEXT,recommended_route TEXT,created_at TEXT DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS incidents(id INTEGER PRIMARY KEY AUTOINCREMENT,type TEXT,description TEXT,latitude REAL,longitude REAL,severity TEXT,status TEXT,location TEXT,reported TEXT);
    CREATE TABLE IF NOT EXISTS field_reports(id INTEGER PRIMARY KEY AUTOINCREMENT,reporter TEXT,location TEXT,report_type TEXT,severity TEXT,description TEXT,latitude REAL,longitude REAL,photo_path TEXT,status TEXT,time TEXT);
    ''')
    if not cur.execute('SELECT 1 FROM users WHERE email=?',('admin@nerlogistics.in',)).fetchone():
        cur.execute('INSERT INTO users(name,email,password_hash) VALUES(?,?,?)',('Operations Admin','admin@nerlogistics.in',hashlib.sha256(b'admin123').hexdigest()))
    if cur.execute('SELECT COUNT(*) FROM incidents').fetchone()[0]==0:
        seed=[('Landslide','Minor landslide debris on hill stretch; single-lane passable.',25.58,91.89,'High','Resolved','Shillong corridor','18 min ago'),('Flood','Waterlogging reported near low-lying road section.',26.14,91.74,'Medium','Monitoring','Guwahati outskirts','32 min ago'),('Road blockage','Temporary blockage due to fallen branches.',25.67,94.11,'Medium','Open','Nagaland corridor','46 min ago')]
        cur.executemany('INSERT INTO incidents(type,description,latitude,longitude,severity,status,location,reported) VALUES(?,?,?,?,?,?,?,?)',seed)
    if cur.execute('SELECT COUNT(*) FROM field_reports').fetchone()[0]==0:
        seed=[('Field Unit 07','Bomdila','Road condition','High','Surface damage and debris observed after rainfall.',None,None,None,'Submitted','20 min ago'),('Field Unit 03','Jowai','Traffic','Medium','Slow-moving traffic near construction zone.',None,None,None,'Verified','1 hr ago')]
        cur.executemany('INSERT INTO field_reports(reporter,location,report_type,severity,description,latitude,longitude,photo_path,status,time) VALUES(?,?,?,?,?,?,?,?,?,?)',seed)
    c.commit(); c.close()
init_db()

# ---------------- Prototype data ----------------
REGIONS=[
 {'name':'Assam','risk':42,'hazards':['Flooding','Heavy rain'],'accessibility':82,'disruptions':7,'rainfall':52,'wind_speed':18,'temperature':24},
 {'name':'Arunachal Pradesh','risk':67,'hazards':['Landslide','Steep terrain'],'accessibility':69,'disruptions':11,'rainfall':77,'wind_speed':16,'temperature':24},
 {'name':'Meghalaya','risk':54,'hazards':['Rainfall','Landslide'],'accessibility':76,'disruptions':8,'rainfall':64,'wind_speed':12,'temperature':24},
 {'name':'Manipur','risk':61,'hazards':['Road blockage','Terrain'],'accessibility':71,'disruptions':9,'rainfall':71,'wind_speed':19,'temperature':24},
 {'name':'Mizoram','risk':64,'hazards':['Landslide','Rainfall'],'accessibility':68,'disruptions':10,'rainfall':74,'wind_speed':13,'temperature':24},
 {'name':'Nagaland','risk':58,'hazards':['Terrain','Road condition'],'accessibility':73,'disruptions':8,'rainfall':68,'wind_speed':16,'temperature':24},
 {'name':'Tripura','risk':39,'hazards':['Flooding','Traffic'],'accessibility':84,'disruptions':5,'rainfall':49,'wind_speed':15,'temperature':24},
 {'name':'Sikkim','risk':63,'hazards':['Landslide','Snow/visibility'],'accessibility':70,'disruptions':9,'rainfall':73,'wind_speed':12,'temperature':24},
]
ALERTS=[
 {'id':1,'location':'Bomdila, Arunachal Pradesh','type':'Landslide','severity':'High','time':'18 min ago','impact':'Risk +14 on hill corridors','route':'Guwahati → Itanagar'},
 {'id':2,'location':'Guwahati outskirts, Assam','type':'Heavy rainfall','severity':'Moderate','time':'32 min ago','impact':'Visibility reduced','route':'Guwahati → Shillong'},
 {'id':3,'location':'Jowai, Meghalaya','type':'Road construction','severity':'Moderate','time':'1 hr ago','impact':'+25 min expected','route':'Guwahati → Shillong'},
 {'id':4,'location':'Dimapur, Nagaland','type':'Traffic congestion','severity':'Moderate','time':'46 min ago','impact':'+35 min expected','route':'Guwahati → Kohima'},
 {'id':5,'location':'Silchar corridor, Assam','type':'Flood','severity':'High','time':'2 hrs ago','impact':'Partial blockage risk','route':'Guwahati → Aizawl'},
]
VEHICLES={'Mini Truck':7.5,'Light Commercial Vehicle':6.5,'Heavy Truck':4.0,'Container Truck':3.2,'Emergency Vehicle':8.0}
FLEET=[
 {'id':'NER-MT-014','type':'Mini Truck','status':'ON TIME','updated':'5 min ago','lat':26.1445,'lng':91.7362},
 {'id':'NER-HT-002','type':'Heavy Truck','status':'DELAYED','updated':'8 min ago','lat':25.5788,'lng':91.8933},
 {'id':'NER-4X-009','type':'4x4 Vehicle','status':'AT RISK','updated':'3 min ago','lat':25.6747,'lng':94.1108},
 {'id':'NER-PU-021','type':'Pickup','status':'ON TIME','updated':'4 min ago','lat':24.8170,'lng':93.9368},
 {'id':'NER-MT-031','type':'Mini Truck','status':'DELIVERED','updated':'18 min ago','lat':23.8315,'lng':91.2868},
]

# ---------------- ML prototype ----------------
rng=np.random.default_rng(42); n=900
X=pd.DataFrame({'rainfall':rng.uniform(0,100,n),'temperature':rng.uniform(8,38,n),'weather_severity':rng.integers(0,6,n),'road_quality':rng.uniform(20,100,n),'terrain_slope':rng.uniform(0,90,n),'historical_disruptions':rng.uniform(0,100,n),'traffic':rng.uniform(0,100,n),'accessibility':rng.uniform(20,100,n),'cargo_weight':rng.uniform(0,30,n)})
y=(.28*X.rainfall+2.7*X.weather_severity+.18*(100-X.road_quality)+.22*X.terrain_slope+.14*X.historical_disruptions+.10*X.traffic+.12*(100-X.accessibility)+.35*X.cargo_weight+rng.normal(0,4,n)).clip(0,100)
MODEL=RandomForestRegressor(n_estimators=120,random_state=42,max_depth=10).fit(X,y)

class AnalyzeRequest(BaseModel):
 source:str; destination:str; vehicle_type:str; cargo_type:str; cargo_weight:float=Field(ge=0); priority:str
class RecalculateRequest(BaseModel): routes:list[dict]; disrupted_route_id:str
class LoginRequest(BaseModel): email:str; password:str
class RegisterRequest(BaseModel): name:str; email:str; password:str
class NearbyRequest(BaseModel): place_name:str=''; latitude:Optional[float]=None; longitude:Optional[float]=None; radius_m:int=10000

# ---------------- Geo/routing ----------------
def geocode(q:str):
    r=requests.get('https://nominatim.openstreetmap.org/search',params={'q':q+', India','format':'json','limit':1},headers={'User-Agent':'NER-Logistics-Intelligence-SIH/2.0'},timeout=8)
    r.raise_for_status(); data=r.json()
    if not data: raise HTTPException(400,f'Location not found: {q}')
    return [float(data[0]['lat']),float(data[0]['lon'])]

def fallback_coords(source,dest):
    known={'guwahati':[26.1445,91.7362],'itanagar':[27.0844,93.6053],'shillong':[25.5788,91.8933],'kohima':[25.6751,94.1086],'aizawl':[23.7271,92.7176],'agartala':[23.8315,91.2868],'gangtok':[27.3389,88.6065]}
    s=known.get(source.lower(),[26.1445,91.7362]); d=known.get(dest.lower(),[27.0844,93.6053]); return s,d

def osrm_single(points):
    """Route through the supplied lon/lat waypoints using OSRM road geometry."""
    coords=';'.join(f"{p[1]},{p[0]}" for p in points)
    url=f"https://router.project-osrm.org/route/v1/driving/{coords}"
    r=requests.get(
        url,
        params={'overview':'full','geometries':'geojson','steps':'false'},
        headers={'User-Agent':'NER-Logistics-Intelligence-SIH/2.0'},
        timeout=10
    )
    r.raise_for_status()
    data=r.json()
    if data.get('code')!='Ok' or not data.get('routes'):
        raise ValueError('No route')
    return data['routes'][0]

def geometry_signature(route):
    coords=route.get('geometry',{}).get('coordinates',[])
    if not coords:
        return ''
    # Sample the geometry so routes with meaningfully different road paths do not collapse into duplicates.
    sample=coords[::max(1,len(coords)//12)]
    return tuple((round(c[0],3),round(c[1],3)) for c in sample)

def osrm_routes(s,d):
    """Return up to three genuinely different road routes.

    OSRM sometimes returns only one alternative for a pair of locations. In that
    case, request additional road routes through different geographic corridors
    rather than drawing tiny shifted copies of the same line. Every returned
    geometry is still produced by OSRM from real road data.
    """
    base=osrm_single([s,d])
    candidates=[base]

    # If the public OSRM service exposes alternatives, use them first.
    try:
        coords=f"{s[1]},{s[0]};{d[1]},{d[0]}"
        url=f"https://router.project-osrm.org/route/v1/driving/{coords}"
        rr=requests.get(url,params={'overview':'full','geometries':'geojson','alternatives':'true','steps':'false'},headers={'User-Agent':'NER-Logistics-Intelligence-SIH/2.0'},timeout=10)
        if rr.ok:
            data=rr.json()
            if data.get('code')=='Ok':
                candidates=list(data.get('routes') or [])[:3]
    except Exception:
        pass

    # Build distinct corridor waypoints around the midpoint. The offsets are
    # deliberately large enough to select different roads at this scale.
    if len(candidates)<3:
        lat1,lon1=s; lat2,lon2=d
        dlat=lat2-lat1; dlon=lon2-lon1
        length=max((dlat*dlat+dlon*dlon)**0.5,0.001)
        # Unit perpendicular vector in lat/lon space.
        perp_lat=-dlon/length
        perp_lon=dlat/length
        mid_lat=lat1+dlat*0.48
        mid_lon=lon1+dlon*0.48
        offsets=[0.0,0.35,-0.35,0.60,-0.60]
        for off in offsets[1:]:
            via=[mid_lat+perp_lat*off,mid_lon+perp_lon*off]
            try:
                route=osrm_single([s,via,d])
                candidates.append(route)
            except Exception:
                continue
            if len(candidates)>=5:
                break

    # Remove duplicate/near-identical geometries while preserving the first route.
    unique=[]
    signatures=set()
    for route in candidates:
        sig=geometry_signature(route)
        if sig and sig in signatures:
            continue
        signatures.add(sig)
        unique.append(route)
        if len(unique)==3:
            break

    if len(unique)>=3:
        return unique[:3]

    # Last-resort geographic corridor requests. These are still sent to OSRM,
    # so the final lines follow mapped roads rather than being hand-drawn.
    lat1,lon1=s; lat2,lon2=d
    for off in (0.9,-0.9,1.2,-1.2):
        if len(unique)>=3: break
        dlat=lat2-lat1; dlon=lon2-lon1; length=max((dlat*dlat+dlon*dlon)**0.5,0.001)
        via=[lat1+dlat*0.45-dlon/length*off,lon1+dlon*0.45+dlat/length*off]
        try:
            route=osrm_single([s,via,d])
            sig=geometry_signature(route)
            if sig and sig not in signatures:
                signatures.add(sig); unique.append(route)
        except Exception:
            continue

    if len(unique)<3:
        raise ValueError('OSRM did not return three distinct road routes')
    return unique[:3]

def make_route_objects(source,destination,vehicle,cargo_weight):
    try:s,d=geocode(source),geocode(destination)
    except Exception:s,d=fallback_coords(source,destination)
    try:rr=osrm_routes(s,d)
    except Exception:
        # Real map remains OpenStreetMap; fallback geometry is geographic and used only if routing service is unavailable.
        rr=[]
        for idx in range(3):
            pts=[]
            for j in range(9):
                t=j/8; lat=s[0]*(1-t)+d[0]*t + (0.25*math.sin(t*math.pi)*(idx-1)); lon=s[1]*(1-t)+d[1]*t + (0.35*math.sin(t*math.pi)*(idx-1)); pts.append([lat,lon])
            rr.append({'distance':0,'duration':0,'geometry':{'coordinates':[[p[1],p[0]] for p in pts]}})
    out=[]
    for idx,r in enumerate(rr):
        distance=r.get('distance',0)/1000 or 300+idx*25; time_min=round((r.get('duration',0)/60) or distance/45*60)
        rain=[52,68,77][idx%3]; terrain=[42,66,74][idx%3]; road=[84,69,62][idx%3]; traffic=[32,45,28][idx%3]; historical=[18,38,50][idx%3]
        points=[[c[1],c[0]] for c in r['geometry']['coordinates']]
        out.append({'id':chr(65+idx),'name':f"Route {chr(65+idx)} — {['Main Corridor','Alternate Corridor','Lower Risk Alternate'][idx]}",'distance':round(distance,1),'time_min':time_min,'road':road,'accessibility':max(20,round(road-idx*4)),'reliability':max(20,round(road-idx*5)),'terrain':terrain,'rainfall':rain,'traffic':traffic,'historical':historical,'weather_severity':max(1,min(5,round(rain/20))),'points':points})
    return out,s,d

def model_predict(route,weight,cargo):
    cargo_factor={'Medicine':-2,'Food':1,'Perishable Goods':3,'Construction Material':4,'General Goods':0}.get(cargo,0)
    row=pd.DataFrame([{'rainfall':route['rainfall'],'temperature':24,'weather_severity':route['weather_severity'],'road_quality':route['road'],'terrain_slope':route['terrain'],'historical_disruptions':route['historical'],'traffic':route['traffic'],'accessibility':route['accessibility'],'cargo_weight':weight}])
    score=float(np.clip(MODEL.predict(row)[0]+cargo_factor,0,100))
    breakdown={'Weather':round(route['rainfall']*.25,1),'Road condition':round((100-route['road'])*.20,1),'Terrain':round(route['terrain']*.18,1),'Historical disruption':round(route['historical']*.15,1),'Traffic':round(route['traffic']*.10,1)}; breakdown['Other']=round(max(0,100-sum(breakdown.values())),1)
    contributors={'Heavy rainfall':round(route['rainfall']*.20,1),'Terrain':round(route['terrain']*.16,1),'Road condition':round((100-route['road'])*.20,1),'Historical disruptions':round(route['historical']*.12,1)}
    return score,breakdown,contributors

def risk_cat(s):return 'Low' if s<=30 else 'Moderate' if s<=60 else 'High' if s<=80 else 'Critical'
def norm(vals,invert=False):
    a=np.array(vals,float); lo,hi=a.min(),a.max(); out=np.ones(len(a))*.5 if hi-lo<1e-9 else (a-lo)/(hi-lo); return 1-out if invert else out

def analyze(req):
    if not req.source.strip() or not req.destination.strip():raise HTTPException(400,'Source and destination are required.')
    if req.vehicle_type not in VEHICLES:raise HTTPException(400,'Please select a supported vehicle type.')
    if req.priority not in ['Fastest','Cheapest','Safest','Balanced']:raise HTTPException(400,'Invalid priority.')
    routes,sc,dc=make_route_objects(req.source.strip(),req.destination.strip(),req.vehicle_type,req.cargo_weight)
    scored=[]
    for r in routes:
        risk,breakdown,contributors=model_predict(r,req.cargo_weight,req.cargo_type); eff=VEHICLES[req.vehicle_type]; fuel=(r['distance']/eff)*96; ops=1200+(r['distance']*4)+max(0,req.cargo_weight)*140; total=fuel+ops+800
        scored.append({**r,'risk':round(risk,1),'risk_category':risk_cat(risk),'cost':{'fuel':round(fuel),'tolls':800,'operations':round(ops),'total':round(total)},'contributors':contributors,'breakdown':breakdown,'weather':{'rainfall':r['rainfall'],'wind_speed':12+r['traffic']%9,'temperature':24,'severity':r['weather_severity'],'condition':'Heavy rain' if r['weather_severity']>=4 else 'Intermittent rain'}})
    weights={'Fastest':[.50,.15,.20,.10,.05],'Cheapest':[.20,.50,.15,.10,.05],'Safest':[.10,.05,.50,.15,.20],'Balanced':[.25,.20,.30,.15,.10]}[req.priority]
    t=norm([r['time_min'] for r in scored],True); c=norm([r['cost']['total'] for r in scored],True); risk=norm([r['risk'] for r in scored],True); a=norm([r['accessibility'] for r in scored]); rel=norm([r['reliability'] for r in scored])
    for i,r in enumerate(scored):r['route_score']=round((weights[0]*t[i]+weights[1]*c[i]+weights[2]*risk[i]+weights[3]*a[i]+weights[4]*rel[i])*100,1);r['estimated_time']=f"{r['time_min']//60}h {r['time_min']%60:02d}m"
    scored.sort(key=lambda x:x['route_score'],reverse=True); rec=scored[0]
    explanation=f"{rec['name']} is recommended because it has {risk_cat(rec['risk']).lower()} predicted risk, accessibility {rec['accessibility']}/100 and reliability {rec['reliability']}/100 while balancing time and estimated cost for the selected priority."
    return {'recommended':rec,'routes':scored,'explanation':explanation,'source_coords':sc,'destination_coords':dc}

# ---------------- Auth ----------------
@app.post('/api/auth/login')
def login(req:LoginRequest):
    row=db().execute('SELECT id,name,email,password_hash FROM users WHERE lower(email)=lower(?)',(req.email.strip(),)).fetchone()
    if not row or row['password_hash']!=hashlib.sha256(req.password.encode()).hexdigest():raise HTTPException(401,'Invalid email or password.')
    return {'user':{'id':row['id'],'name':row['name'],'email':row['email']}}
@app.post('/api/auth/register')
def register(req:RegisterRequest):
    c=db()
    try:c.execute('INSERT INTO users(name,email,password_hash) VALUES(?,?,?)',(req.name.strip(),req.email.strip(),hashlib.sha256(req.password.encode()).hexdigest()));c.commit();row=c.execute('SELECT id,name,email FROM users WHERE email=?',(req.email.strip(),)).fetchone();return {'user':dict(row)}
    except sqlite3.IntegrityError:raise HTTPException(409,'An account with this email already exists.')
    finally:c.close()

@app.get('/api/alerts')
def alerts():return ALERTS
@app.get('/api/regions')
def regions():return REGIONS
@app.get('/api/weather')
def weather():return {'regions':REGIONS}
@app.get('/api/vehicles')
def vehicles():return FLEET
@app.get('/api/incidents')
def incidents():return [dict(x) for x in db().execute('SELECT * FROM incidents ORDER BY id DESC').fetchall()]
@app.get('/api/routes')
def route_catalog():return {'message':'Routes are generated dynamically from source and destination using OpenStreetMap/OSRM.'}

@app.post('/api/routes/analyze')
def route_analyze(req:AnalyzeRequest):
    data=analyze(req); c=db();c.execute('INSERT INTO route_analyses(source,destination,vehicle_type,cargo_type,cargo_weight,priority,recommended_route) VALUES(?,?,?,?,?,?,?)',(req.source,req.destination,req.vehicle_type,req.cargo_type,req.cargo_weight,req.priority,data['recommended']['name']));c.commit();c.close();return data
@app.post('/api/routes/recalculate')
def recalculate(req:RecalculateRequest):
    routes=[dict(x) for x in req.routes]
    for r in routes:
        if r['id']==req.disrupted_route_id:r['risk']=min(100,round(r['risk']+26,1));r['reliability']=max(0,r['reliability']-18);r['risk_category']=risk_cat(r['risk'])
    routes.sort(key=lambda r:(r['risk'],-r['reliability'],-r.get('route_score',0)));new=routes[0]
    return {'routes':routes,'recommended':new,'message':f"{new['name']} is recommended because the previous route now has elevated disruption risk."}

@app.get('/api/field-reports')
def field_reports():
    return [dict(x) for x in db().execute('SELECT id,reporter,location,report_type as type,severity,description,latitude,longitude,photo_path as photo_url,status,time FROM field_reports ORDER BY id DESC').fetchall()]
@app.post('/api/field-reports')
async def create_field_report(reporter:str=Form(...),location:str=Form(...),report_type:str=Form(...),severity:str=Form(...),description:str=Form(''),latitude:Optional[float]=Form(None),longitude:Optional[float]=Form(None),photo:Optional[UploadFile]=File(None)):
    path=None
    if photo and photo.filename:
        ext=os.path.splitext(photo.filename)[1].lower() or '.jpg'; name=f"{uuid.uuid4().hex}{ext}"; path=f'/uploads/{name}';
        with open(os.path.join(UPLOADS,name),'wb') as f:f.write(await photo.read())
    c=db();c.execute('INSERT INTO field_reports(reporter,location,report_type,severity,description,latitude,longitude,photo_path,status,time) VALUES(?,?,?,?,?,?,?,?,?,?)',(reporter,location,report_type,severity,description,latitude,longitude,path,'Submitted','just now'));c.commit();rid=c.execute('SELECT last_insert_rowid()').fetchone()[0];row=c.execute('SELECT id,reporter,location,report_type as type,severity,description,latitude,longitude,photo_path as photo_url,status,time FROM field_reports WHERE id=?',(rid,)).fetchone();c.close();return dict(row)

@app.post('/api/emergency/nearby')
def nearby(req:NearbyRequest):
    lat, lon = req.latitude, req.longitude
    if lat is None or lon is None:
        if not req.place_name.strip(): raise HTTPException(400,'Enter a place name or latitude/longitude.')
        try: lat,lon=geocode(req.place_name.strip())
        except Exception: raise HTTPException(400,'Place name could not be found. Enter a valid place name or latitude and longitude.')
    try: radius=max(1,int(req.radius_m))
    except Exception: raise HTTPException(400,'Search radius must be a valid number.')
    lat=float(lat); lon=float(lon)
    categories={'hospital':'Hospital','police':'Police station','fuel':'Fuel station','garage':'Garage'}
    def distance_m(a,b,c,d):
        R=6371000.0; p1=math.radians(a); p2=math.radians(c); dp=math.radians(c-a); dl=math.radians(d-b)
        h=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
        return 2*R*math.asin(min(1,math.sqrt(h)))
    def normalize(lat0,lon0,name,category,address=''):
        if lat0 is None or lon0 is None:return None
        d=distance_m(lat,lon,float(lat0),float(lon0))
        if d>radius:return None
        return {'category':category,'name':name or categories[category],'distance_m':round(d),'latitude':float(lat0),'longitude':float(lon0),'address':address or 'OpenStreetMap place'}
    def normalize_element(el):
        tags=el.get('tags',{}); plat=el.get('lat') or el.get('center',{}).get('lat'); plon=el.get('lon') or el.get('center',{}).get('lon')
        if plat is None or plon is None:return None
        amen=tags.get('amenity',''); shop=tags.get('shop',''); craft=tags.get('craft','')
        if amen=='hospital':cat='hospital'
        elif amen=='police':cat='police'
        elif amen=='fuel':cat='fuel'
        elif shop=='car_repair' or craft=='car_repair':cat='garage'
        else:return None
        addr=', '.join(x for x in [tags.get('addr:housenumber'),tags.get('addr:street'),tags.get('addr:city')] if x)
        return normalize(plat,plon,tags.get('name') or tags.get('official_name'),cat,addr)
    found=[]
    query=(f'[out:json][timeout:20];('
           f'nwr[amenity=hospital](around:{radius},{lat},{lon});'
           f'nwr[amenity=police](around:{radius},{lat},{lon});'
           f'nwr[amenity=fuel](around:{radius},{lat},{lon});'
           f'nwr[shop=car_repair](around:{radius},{lat},{lon});'
           f'nwr[craft=car_repair](around:{radius},{lat},{lon});'
           f');out center tags;')
    for endpoint in ['https://overpass-api.de/api/interpreter','https://overpass.kumi.systems/api/interpreter','https://overpass.private.coffee/api/interpreter']:
        try:
            rr=requests.post(endpoint,data=query,timeout=18,headers={'User-Agent':'NER-Logistics-Intelligence-SIH/2.0'}); rr.raise_for_status(); payload=rr.json()
            found=[x for el in payload.get('elements',[]) if (x:=normalize_element(el))]
            if found:break
        except Exception:continue
    if not found:
        delta_lat=radius/111000.0; delta_lon=radius/(111000.0*max(0.2,math.cos(math.radians(lat))))
        viewbox=f'{lon-delta_lon},{lat+delta_lat},{lon+delta_lon},{lat-delta_lat}'
        for cat,q in {'hospital':'hospital','police':'police station','fuel':'fuel station','garage':'car repair'}.items():
            try:
                rr=requests.get('https://nominatim.openstreetmap.org/search',params={'q':f'{q}, {req.place_name.strip()}' if req.place_name.strip() else q,'format':'jsonv2','limit':10,'viewbox':viewbox,'bounded':1},headers={'User-Agent':'NER-Logistics-Intelligence-SIH/2.0'},timeout=12); rr.raise_for_status()
                for el in rr.json():
                    item=normalize(el.get('lat'),el.get('lon'),el.get('display_name','').split(',')[0],cat,el.get('display_name',''))
                    if item:found.append(item)
            except Exception:continue
    if not found:
        fallback={
          'Shillong':{'hospital':(25.5780,91.8930,'Shillong Civil Hospital'),'fuel':(25.5758,91.8790,'IndianOil Fuel Station Shillong'),'garage':(25.5840,91.8915,'Shillong Auto Service Centre'),'police':(25.5748,91.8895,'Shillong Police Station')},
          'Guwahati':{'hospital':(26.1448,91.7365,'Gauhati Medical College Hospital'),'fuel':(26.1420,91.7440,'IndianOil Fuel Station Guwahati'),'garage':(26.1510,91.7310,'Guwahati Auto Service Centre'),'police':(26.1450,91.7390,'Guwahati Police Station')},
          'Itanagar':{'hospital':(27.0844,93.6053,'Tomo Riba Institute of Health & Medical Sciences'),'fuel':(27.1020,93.6160,'Itanagar Fuel Station'),'garage':(27.0910,93.6030,'Itanagar Auto Service Centre'),'police':(27.0840,93.6100,'Itanagar Police Station')},
          'Kohima':{'hospital':(25.6751,94.1086,'Naga Hospital Authority Kohima'),'fuel':(25.6670,94.1100,'Kohima Fuel Station'),'garage':(25.6800,94.1120,'Kohima Auto Service Centre'),'police':(25.6700,94.1050,'Kohima Police Station')},
          'Aizawl':{'hospital':(23.7271,92.7176,'Civil Hospital Aizawl'),'fuel':(23.7300,92.7150,'Aizawl Fuel Station'),'garage':(23.7240,92.7200,'Aizawl Auto Service Centre'),'police':(23.7280,92.7190,'Aizawl Police Station')},
          'Agartala':{'hospital':(23.8315,91.2868,'Agartala Government Medical College'),'fuel':(23.8350,91.2900,'Agartala Fuel Station'),'garage':(23.8280,91.2830,'Agartala Auto Service Centre'),'police':(23.8330,91.2870,'Agartala Police Station')},
          'Gangtok':{'hospital':(27.3389,88.6065,'STNM Hospital Gangtok'),'fuel':(27.3300,88.6100,'Gangtok Fuel Station'),'garage':(27.3420,88.6030,'Gangtok Auto Service Centre'),'police':(27.3350,88.6070,'Gangtok Police Station')}}
        key=None; low=req.place_name.strip().lower()
        for city in fallback:
            if city.lower() in low or low in city.lower():key=city;break
        if key is None:key=min(fallback,key=lambda city:distance_m(lat,lon,*fallback[city]['hospital'][:2]))
        for cat,(plat,plon,name) in fallback[key].items():
            item=normalize(plat,plon,name,cat,key+' • prototype fallback')
            if item:found.append(item)
    best={}
    for item in sorted(found,key=lambda x:x['distance_m']):best.setdefault(item['category'],item)
    source='OpenStreetMap' if found and not any('prototype fallback' in x.get('address','') for x in found) else 'Prototype fallback'
    return {'origin':{'place_name':req.place_name,'latitude':lat,'longitude':lon},'places':list(best.values()),'source':source}

@app.post('/api/risk/predict')
def predict(req:AnalyzeRequest):
    data=analyze(req);r=data['recommended'];return {'risk_score':r['risk'],'category':r['risk_category'],'main_risk_factors':r['contributors'],'breakdown':r['breakdown'],'confidence':84}
@app.post('/api/cost/calculate')
def calculate_cost(req:AnalyzeRequest):return analyze(req)['recommended']['cost']
@app.get('/api/health')
def health():return {'status':'ok','database':os.path.exists(DB),'ml':'RandomForestRegressor','maps':'OpenStreetMap + OSRM'}

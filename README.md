# NER Logistics Intelligence — SIH Round 2 Prototype

This version preserves the original DM Sans / Space Grotesk UI, colors, cards and layout while adding only the requested Round 2 functionality.

## Run in VS Code

### Terminal 1 — Backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

### Terminal 2 — Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in Terminal 2.

## Demo login

Email: `admin@nerlogistics.in`

Password: `admin123`

## Maps

The prototype uses OpenStreetMap tiles and OSRM road routing. Internet access is required for live map tiles, road routing and nearby emergency-place lookup. If OSRM is temporarily unavailable, the backend falls back to geographic route geometry so the application still opens.

## Database

SQLite is created automatically as `backend/ner_logistics.db` on first backend start. It stores users, route analyses, incidents and field reports. Uploaded field-report photos are stored in `backend/uploads/`.

## Notes

- No vehicle maximum-capacity validation is hard-coded.
- Cargo weight accepts arbitrary non-negative numeric input.
- Route Planner supports cargo type including a custom `Other` option.
- Selecting Route A/B/C changes the highlighted route on the real map.
- Live Map displays stored vehicle coordinates and incident coordinates as real map markers.

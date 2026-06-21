"use client";
import { useEffect, useRef, useState } from "react";

interface Props {
  latitude: number;
  longitude: number;
  location: string;
  confidence: number;
}

const TILE_PROVIDERS = {
  street: {
    url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    label: "지도",
    attr: "© OpenStreetMap",
  },
  satellite: {
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    label: "위성",
    attr: "© Esri",
  },
  topo: {
    url: "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
    label: "지형",
    attr: "© OpenTopoMap",
  },
};

type TileKey = keyof typeof TILE_PROVIDERS;

export default function MapView({ latitude, longitude, location, confidence }: Props) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstance = useRef<any>(null);
  const tileLayerRef = useRef<any>(null);
  const [tileKey, setTileKey] = useState<TileKey>("street");
  const [copied, setCopied] = useState(false);

  const confColor = confidence >= 0.9 ? "#10B981" : confidence >= 0.7 ? "#F59E0B" : "#F43F5E";
  const radiusM = confidence >= 0.9 ? 50 : confidence >= 0.7 ? 200 : 800;

  const copyCoords = () => {
    navigator.clipboard.writeText(`${latitude.toFixed(6)}, ${longitude.toFixed(6)}`);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  // 지도 초기화
  useEffect(() => {
    if (!mapRef.current || !latitude || !longitude) return;
    if (mapInstance.current) return;
    // Leaflet이 이미 초기화한 DOM인 경우 스킵 (StrictMode 이중 실행 방어)
    if ((mapRef.current as any)._leaflet_id) return;

    let mounted = true;

    const init = async () => {
      const L = (await import("leaflet")).default;
      await import("leaflet/dist/leaflet.css");

      if (!mounted || !mapRef.current) return;
      if ((mapRef.current as any)._leaflet_id) return;

      const map = L.map(mapRef.current!, {
        center: [latitude, longitude],
        zoom: 15,
        zoomControl: false,
        attributionControl: false,
      });

      // 커스텀 줌 컨트롤 (우측 하단)
      L.control.zoom({ position: "bottomright" }).addTo(map);
      L.control.attribution({ position: "bottomright", prefix: false }).addTo(map);

      // 기본 타일
      const tile = L.tileLayer(TILE_PROVIDERS.street.url, {
        attribution: TILE_PROVIDERS.street.attr,
        maxZoom: 19,
      });
      tile.addTo(map);
      tileLayerRef.current = tile;

      // 불확실성 원
      L.circle([latitude, longitude], {
        radius: radiusM,
        color: confColor,
        fillColor: confColor,
        fillOpacity: 0.1,
        weight: 2,
        dashArray: confidence < 0.7 ? "6 4" : undefined,
      }).addTo(map);

      // 마커
      const icon = L.divIcon({
        html: `<div style="
          position:relative;width:20px;height:20px;
        ">
          <div style="
            position:absolute;inset:0;border-radius:50%;
            background:${confColor};opacity:0.3;
            animation:ping 1.5s ease-out infinite;
          "></div>
          <div style="
            position:absolute;inset:4px;border-radius:50%;
            background:${confColor};border:2px solid white;
            box-shadow:0 0 8px ${confColor};
          "></div>
        </div>`,
        className: "",
        iconSize: [20, 20],
        iconAnchor: [10, 10],
      });

      L.marker([latitude, longitude], { icon })
        .addTo(map)
        .bindPopup(
          `<div style="font-family:monospace;font-size:12px;min-width:180px">
            <strong style="font-size:13px">${location}</strong><br/>
            <span style="color:${confColor}">신뢰도: ${(confidence * 100).toFixed(1)}%</span><br/>
            <span style="color:#888">${latitude.toFixed(6)}, ${longitude.toFixed(6)}</span><br/>
            <a href="https://maps.google.com/?q=${latitude},${longitude}" target="_blank"
               style="color:#0EA5E9;text-decoration:none">Google Maps ↗</a>
          </div>`,
          { closeButton: false }
        )
        .openPopup();

      // 스타일 주입 (ping animation)
      if (!document.getElementById("leaflet-ping-style")) {
        const style = document.createElement("style");
        style.id = "leaflet-ping-style";
        style.textContent = `@keyframes ping{0%{transform:scale(1);opacity:.3}70%,100%{transform:scale(2.5);opacity:0}}`;
        document.head.appendChild(style);
      }

      mapInstance.current = { map, L };
    };

    init().catch(console.error);

    return () => {
      mounted = false;
      if (mapInstance.current) {
        mapInstance.current.map.remove();
        mapInstance.current = null;
      }
    };
  }, [latitude, longitude, location, confidence]);

  // 타일 레이어 전환
  useEffect(() => {
    if (!mapInstance.current) return;
    const { map, L } = mapInstance.current;
    if (tileLayerRef.current) {
      map.removeLayer(tileLayerRef.current);
    }
    const provider = TILE_PROVIDERS[tileKey];
    const newTile = L.tileLayer(provider.url, {
      attribution: provider.attr,
      maxZoom: 19,
    });
    newTile.addTo(map);
    tileLayerRef.current = newTile;
  }, [tileKey]);

  return (
    <div className="relative rounded-xl overflow-hidden border border-[#1E2D45]" style={{ height: 280 }}>
      <div ref={mapRef} style={{ height: "100%", width: "100%", background: "#0D1220" }} />

      {/* 좌표 복사 배지 */}
      <button
        onClick={copyCoords}
        className="absolute top-2 left-2 bg-black/75 hover:bg-black/90 transition-colors
          text-xs font-mono px-2 py-1 rounded flex items-center gap-1.5 z-[1000]"
        style={{ color: confColor }}
      >
        {latitude.toFixed(5)}, {longitude.toFixed(5)}
        <span className="text-[#475569]">{copied ? "✓" : "⎘"}</span>
      </button>

      {/* 타일 전환 버튼 */}
      <div className="absolute top-2 right-2 flex gap-1 z-[1000]">
        {(Object.keys(TILE_PROVIDERS) as TileKey[]).map((k) => (
          <button
            key={k}
            onClick={() => setTileKey(k)}
            className={`text-xs px-2 py-1 rounded transition-colors font-mono ${
              tileKey === k
                ? "bg-[#0EA5E9] text-black"
                : "bg-black/70 text-[#64748B] hover:text-white"
            }`}
          >
            {TILE_PROVIDERS[k].label}
          </button>
        ))}
      </div>

      {/* 신뢰도 레이블 */}
      <div className="absolute bottom-8 left-2 z-[1000]">
        <div className="bg-black/75 px-2 py-0.5 rounded text-xs font-bold"
             style={{ color: confColor }}>
          {confidence >= 0.9 ? "HIGH" : confidence >= 0.7 ? "MEDIUM" : "LOW"}&nbsp;
          {(confidence * 100).toFixed(0)}%
        </div>
      </div>
    </div>
  );
}

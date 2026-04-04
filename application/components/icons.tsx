type IconProps = {
  className?: string;
  stroke?: string;
};

function base(className?: string) {
  return {
    viewBox: "0 0 24 24",
    fill: "none",
    strokeWidth: 1.9,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    className
  };
}

export function LeafIcon({ className, stroke = "currentColor" }: IconProps) {
  return (
    <svg {...base(className)} stroke={stroke}>
      <path d="M19.5 4.5c-5.9-.7-10.4.8-13.1 3.5-3.6 3.6-3.5 8.8-.2 12 3.2 3.2 8.4 3.4 12-.2 2.7-2.7 4.2-7.2 3.5-13.1-.1-.8-1.3-2-2.2-2.2Z" />
      <path d="M8.5 15.5c1.2-2.9 3.6-5.3 7-7" />
    </svg>
  );
}

export function ArrowLeftIcon({ className, stroke = "currentColor" }: IconProps) {
  return (
    <svg {...base(className)} stroke={stroke}>
      <path d="M19 12H5" />
      <path d="m12 19-7-7 7-7" />
    </svg>
  );
}

export function ArrowRightIcon({ className, stroke = "currentColor" }: IconProps) {
  return (
    <svg {...base(className)} stroke={stroke}>
      <path d="M5 12h14" />
      <path d="m12 5 7 7-7 7" />
    </svg>
  );
}

export function DashboardIcon({ className, stroke = "currentColor" }: IconProps) {
  return (
    <svg {...base(className)} stroke={stroke}>
      <rect x="4" y="4" width="6" height="6" />
      <rect x="14" y="4" width="6" height="6" />
      <rect x="4" y="14" width="6" height="6" />
      <rect x="14" y="14" width="6" height="6" />
    </svg>
  );
}

export function TrendIcon({ className, stroke = "currentColor" }: IconProps) {
  return (
    <svg {...base(className)} stroke={stroke}>
      <path d="M4 18V6" />
      <path d="M10 18V10" />
      <path d="M16 18V13" />
      <path d="M22 18V4" />
      <path d="m3 13 5-5 4 3 8-7" />
    </svg>
  );
}

export function UpRightIcon({ className, stroke = "currentColor" }: IconProps) {
  return (
    <svg {...base(className)} stroke={stroke}>
      <path d="m7 17 10-10" />
      <path d="M8 7h9v9" />
    </svg>
  );
}

export function AvocadoIcon({ className, stroke = "currentColor" }: IconProps) {
  return (
    <svg {...base(className)} stroke={stroke}>
      <path d="M12 4c-1.7 1.7-4.6 5.3-4.6 8.9A4.6 4.6 0 0 0 12 17.5a4.6 4.6 0 0 0 4.6-4.6C16.6 9.3 13.7 5.7 12 4Z" />
      <circle cx="12" cy="13" r="1.8" />
      <path d="M13.7 5.8c1.1-.8 2.1-1 3.3-.8" />
    </svg>
  );
}

export function TomatoIcon({ className, stroke = "currentColor" }: IconProps) {
  return (
    <svg {...base(className)} stroke={stroke}>
      <circle cx="12" cy="13" r="6.5" />
      <path d="M12 6.5V3.5" />
      <path d="m12 6.5 2.5-2" />
      <path d="m12 6.5-2.5-2" />
      <path d="m12 6.5 3.7-.8" />
      <path d="m12 6.5-3.7-.8" />
    </svg>
  );
}

export function WeatherIcon({ className, stroke = "currentColor" }: IconProps) {
  return (
    <svg {...base(className)} stroke={stroke}>
      <path d="M12 3v10" />
      <path d="M9 7.5a3 3 0 1 1 6 0v6.8a4.5 4.5 0 1 1-6 0Z" />
      <path d="M17 7h4" />
      <path d="M17 11h2.5" />
    </svg>
  );
}

export function ExchangeIcon({ className, stroke = "currentColor" }: IconProps) {
  return (
    <svg {...base(className)} stroke={stroke}>
      <path d="M3 8h13" />
      <path d="m12 4 4 4-4 4" />
      <path d="M21 16H8" />
      <path d="m12 12-4 4 4 4" />
      <circle cx="12" cy="12" r="9" />
    </svg>
  );
}

export function GasIcon({ className, stroke = "currentColor" }: IconProps) {
  return (
    <svg {...base(className)} stroke={stroke}>
      <path d="M7 20V6.5A2.5 2.5 0 0 1 9.5 4h3A2.5 2.5 0 0 1 15 6.5V20" />
      <path d="M7 11h8" />
      <path d="M15 8h2l2 2v6a2 2 0 1 0 4 0v-3" />
    </svg>
  );
}

export function BoxIcon({ className, stroke = "currentColor" }: IconProps) {
  return (
    <svg {...base(className)} stroke={stroke}>
      <path d="M4 7h16v13H4z" />
      <path d="M4 11h16" />
      <path d="m9 7 1.5-3h3L15 7" />
    </svg>
  );
}

export function AnalyticsIcon({ className, stroke = "currentColor" }: IconProps) {
  return (
    <svg {...base(className)} stroke={stroke}>
      <path d="m4 16 4-4 4 3 5-7 3 2" />
      <circle cx="17" cy="17" r="3" />
      <path d="m19.2 19.2 2.3 2.3" />
    </svg>
  );
}

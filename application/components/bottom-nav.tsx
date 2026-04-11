"use client";

import Link from "next/link";
import type { Route } from "next";
import type { ComponentType } from "react";
import { usePathname } from "next/navigation";
import { AvocadoIcon, DashboardIcon, TomatoIcon } from "@/components/icons";

const navItems: Array<{
  href: Route;
  label: string;
  icon: ComponentType<{ className?: string; stroke?: string }>;
}> = [
  { href: "/", label: "Dashboard", icon: DashboardIcon },
  { href: "/avocado", label: "Avocado", icon: AvocadoIcon },
  { href: "/tomato", label: "Tomato", icon: TomatoIcon }
];

export function BottomNav() {
  const pathname = usePathname();

  return (
    <>
      <p className="app-note">
        Forecast accuracy varies by product. Avocado results should be interpreted with
        caution.
      </p>
      <nav className="bottom-nav">
        {navItems.map(({ href, label, icon: Icon }) => {
          const isActive = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className={`bottom-nav__item${isActive ? " is-active" : ""}`}
            >
              <Icon className="bottom-nav__icon" />
              <span>{label}</span>
            </Link>
          );
        })}
      </nav>
    </>
  );
}

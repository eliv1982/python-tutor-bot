import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

import { setUnauthorizedHandler } from "../api/client";

afterEach(() => {
  cleanup();
  setUnauthorizedHandler(null);
  // Expire every cookie a test may have set, on the same (https) origin.
  for (const cookie of document.cookie.split(";")) {
    const name = cookie.split("=")[0]?.trim();
    if (name) {
      document.cookie = `${name}=; Max-Age=0; Path=/`;
      document.cookie = `${name}=; Max-Age=0; Path=/; Secure`;
    }
  }
});

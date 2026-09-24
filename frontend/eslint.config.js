import js from "@eslint/js";
import { defineConfig, globalIgnores } from "eslint/config";
import reactHooks from "eslint-plugin-react-hooks";
import globals from "globals";
import tseslint from "typescript-eslint";

export default defineConfig([
  globalIgnores(["dist", "node_modules"]),
  {
    files: ["**/*.{ts,tsx}"],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommendedTypeChecked,
      reactHooks.configs.flat.recommended,
    ],
    languageOptions: {
      ecmaVersion: 2023,
      globals: globals.browser,
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      // Stage 7B-1 browser-storage policy: no authentication material (or
      // anything else) is persisted client-side.
      "no-restricted-globals": [
        "error",
        { name: "localStorage", message: "Browser storage is not used (Stage 7B-1 policy)." },
        { name: "sessionStorage", message: "Browser storage is not used (Stage 7B-1 policy)." },
        { name: "indexedDB", message: "Browser storage is not used (Stage 7B-1 policy)." },
      ],
      "no-restricted-properties": [
        "error",
        { object: "window", property: "localStorage", message: "Browser storage is not used." },
        { object: "window", property: "sessionStorage", message: "Browser storage is not used." },
        { object: "window", property: "indexedDB", message: "Browser storage is not used." },
        { object: "globalThis", property: "localStorage", message: "Browser storage is not used." },
        { object: "globalThis", property: "sessionStorage", message: "Browser storage is not used." },
        { object: "globalThis", property: "indexedDB", message: "Browser storage is not used." },
      ],
      // Server-provided text is only ever rendered as React text nodes.
      "no-restricted-syntax": [
        "error",
        {
          selector: "JSXAttribute[name.name='dangerouslySetInnerHTML']",
          message: "Raw HTML rendering is forbidden; render text through React.",
        },
        {
          selector: "MemberExpression[property.name=/^(innerHTML|outerHTML)$/]",
          message: "Raw HTML assignment is forbidden; render text through React.",
        },
      ],
    },
  },
  {
    // Tests deliberately touch the storage APIs to prove they are unused.
    files: ["**/*.test.{ts,tsx}"],
    rules: {
      "no-restricted-globals": "off",
      "no-restricted-properties": "off",
      "no-restricted-syntax": "off",
    },
  },
]);

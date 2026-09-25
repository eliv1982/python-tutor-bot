import { describe, expect, it } from "vitest";

import { jsonResponse, mockFetch } from "../test/http";
import { ApiError, UNEXPECTED_RESPONSE_DETAIL } from "./client";
import {
  isSafeTelegramDeepLink,
  isTelegramLinkStartResponse,
  startTelegramLink,
  unlinkGithub,
} from "./accountLinking";

const SECRET = "A".repeat(43);
const BOT_PATH = "/my_tutor_bot";
const DEEP_LINK = `https://t.me${BOT_PATH}?start=link_${SECRET}`;
const RESPONSE = { deep_link: DEEP_LINK, bot_path: BOT_PATH, expires_at: "2026-09-25T12:00:00Z" };

describe("Telegram deep-link validation", () => {
  it("accepts the exact backend URL and response shape", () => {
    expect(isSafeTelegramDeepLink(DEEP_LINK, BOT_PATH)).toBe(true);
    expect(isTelegramLinkStartResponse(RESPONSE)).toBe(true);
  });

  it.each([
    `http://t.me${BOT_PATH}?start=link_${SECRET}`,
    `https://evil.example${BOT_PATH}?start=link_${SECRET}`,
    `https://t.me.evil.example${BOT_PATH}?start=link_${SECRET}`,
    `https://ｔ.ｍｅ${BOT_PATH}?start=link_${SECRET}`,
    `https://user@t.me${BOT_PATH}?start=link_${SECRET}`,
    `https://t.me:443${BOT_PATH}?start=link_${SECRET}`,
    `https://t.me${BOT_PATH}?start=link_${SECRET}#fragment`,
    `https://t.me/other_bot?start=link_${SECRET}`,
    `https://t.me/%6dy_tutor_bot?start=link_${SECRET}`,
    `https://t.me${BOT_PATH}/extra?start=link_${SECRET}`,
    `https://t.me%2f.evil.example${BOT_PATH}?start=link_${SECRET}`,
    `https://t.me${BOT_PATH}?start=link_${SECRET}&next=evil`,
    `https://t.me${BOT_PATH}?start=link_${SECRET}&start=link_${SECRET}`,
    `https://t.me${BOT_PATH}`,
    `https://t.me${BOT_PATH}?start=`,
    `https://t.me${BOT_PATH}?st%61rt=link_${SECRET}`,
    `https://t.me${BOT_PATH}?start=link_%41${"A".repeat(42)}`,
    `https://t.me${BOT_PATH}?start=other_${SECRET}`,
    `https://t.me${BOT_PATH}?start=link_${"A".repeat(42)}`,
    `https://t.me${BOT_PATH}?start=link_${"A".repeat(42)}+`,
    `https://t.me${BOT_PATH}?start=link_${"A".repeat(42)}B`,
    ` https://t.me${BOT_PATH}?start=link_${SECRET}`,
    `https://t.me${BOT_PATH}?start=link_${SECRET}\n`,
    `https://t.me${BOT_PATH}?start=link_${SECRET}\t`,
  ])("rejects unsafe or noncanonical link %s", (link) => {
    expect(isSafeTelegramDeepLink(link, BOT_PATH)).toBe(false);
  });

  it.each(["/bot", "/1invalid_bot", "/my-tutor-bot", "/my_tutor_bot/extra", "/my_tutor_bot%2f"])(
    "rejects invalid expected bot path %s",
    (botPath) => expect(isSafeTelegramDeepLink(DEEP_LINK, botPath)).toBe(false),
  );

  it("rejects extra response fields and malformed expiration", () => {
    expect(isTelegramLinkStartResponse({ ...RESPONSE, extra: true })).toBe(false);
    expect(isTelegramLinkStartResponse({ ...RESPONSE, expires_at: "not-a-date" })).toBe(false);
    expect(isTelegramLinkStartResponse({ ...RESPONSE, expires_at: "0" })).toBe(false);
    expect(
      isTelegramLinkStartResponse({ deep_link: RESPONSE.deep_link, bot_path: RESPONSE.bot_path }),
    ).toBe(false);
  });
});

describe("account-linking API", () => {
  it("starts linking bodylessly and returns the original backend URL", async () => {
    document.cookie = "csrf_token=dev-token; Path=/";
    const { calls } = mockFetch(() => jsonResponse(200, RESPONSE));

    const result = await startTelegramLink();

    expect(result.deep_link).toBe(DEEP_LINK);
    expect(calls[0]?.url).toBe("/api/link/telegram/start");
    expect(calls[0]?.init.body).toBeUndefined();
    expect(calls[0]?.headers.has("content-type")).toBe(false);
    expect(calls[0]?.headers.get("x-csrf-token")).toBe("dev-token");
  });

  it("rejects an unsafe successful response with a fixed client-owned error", async () => {
    mockFetch(() => jsonResponse(200, { ...RESPONSE, deep_link: `https://evil.example${BOT_PATH}?start=link_${SECRET}` }));

    await expect(startTelegramLink()).rejects.toEqual(new ApiError(200, UNEXPECTED_RESPONSE_DETAIL));
  });

  it("unlinks GitHub bodylessly and validates the fixed acknowledgement", async () => {
    const { calls } = mockFetch(() => jsonResponse(200, { status: "ok" }));

    await expect(unlinkGithub()).resolves.toEqual({ status: "ok" });
    expect(calls[0]?.url).toBe("/api/unlink/github");
    expect(calls[0]?.init.body).toBeUndefined();
  });

  it.each([{ status: "no" }, { status: "ok", extra: true }, {}])(
    "rejects malformed unlink acknowledgement %#",
    async (body) => {
      mockFetch(() => jsonResponse(200, body));
      await expect(unlinkGithub()).rejects.toEqual(new ApiError(200, UNEXPECTED_RESPONSE_DETAIL));
    },
  );
});

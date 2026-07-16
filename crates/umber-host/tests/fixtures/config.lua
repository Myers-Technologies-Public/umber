-- Host ABI v1 Lua fixture exercising config_get: emit a manifest-declared value.
umber.register("cfg.show", "Config: Show", function()
  umber.emit(umber.config_get("greeting") or "none")
end)

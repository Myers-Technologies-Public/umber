-- Host ABI v1 Lua fixture: register a command with a handler that emits text.
umber.register("hello.lua", "Hello: Lua", function()
  umber.emit("hello from lua")
end)

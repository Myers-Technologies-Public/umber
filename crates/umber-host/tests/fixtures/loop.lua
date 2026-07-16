-- Host ABI v1 Lua fixture whose handler spins forever. The instruction-count
-- hook must abort it (HostError::Timeout) within the budget.
umber.register("loop.lua", "Loop: Lua", function()
  while true do end
end)

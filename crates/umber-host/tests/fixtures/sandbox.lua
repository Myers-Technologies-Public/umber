-- Sandbox regression fixture: the host must not expose any ambient-authority
-- stdlib to module scripts. If any of these globals exist, the deny-all
-- sandbox claim is false.
umber.register("sandbox.check", "Sandbox check", function()
    if os ~= nil or io ~= nil or package ~= nil or require ~= nil or debug ~= nil then
        umber.emit("LEAK")
    else
        umber.emit("clean")
    end
end)

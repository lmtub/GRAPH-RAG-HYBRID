void vga_hw_invalidate(void)

{

    if (active_console->hw_invalidate)

        active_console->hw_invalidate(active_console->hw);

}

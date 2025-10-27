static void vga_reset(void *opaque)

{

    VGAState *s = (VGAState *) opaque;



    s->lfb_addr = 0;

    s->lfb_end = 0;

    s->map_addr = 0;

    s->map_end = 0;

    s->lfb_vram_mapped = 0;

    s->bios_offset = 0;

    s->bios_size = 0;

    s->sr_index = 0;

    memset(s->sr, '\0', sizeof(s->sr));

    s->gr_index = 0;

    memset(s->gr, '\0', sizeof(s->gr));

    s->ar_index = 0;

    memset(s->ar, '\0', sizeof(s->ar));

    s->ar_flip_flop = 0;

    s->cr_index = 0;

    memset(s->cr, '\0', sizeof(s->cr));

    s->msr = 0;

    s->fcr = 0;

    s->st00 = 0;

    s->st01 = 0;

    s->dac_state = 0;

    s->dac_sub_index = 0;

    s->dac_read_index = 0;

    s->dac_write_index = 0;

    memset(s->dac_cache, '\0', sizeof(s->dac_cache));

    s->dac_8bit = 0;

    memset(s->palette, '\0', sizeof(s->palette));

    s->bank_offset = 0;

#ifdef CONFIG_BOCHS_VBE

    s->vbe_index = 0;

    memset(s->vbe_regs, '\0', sizeof(s->vbe_regs));

    s->vbe_regs[VBE_DISPI_INDEX_ID] = VBE_DISPI_ID0;

    s->vbe_start_addr = 0;

    s->vbe_line_offset = 0;

    s->vbe_bank_mask = (s->vram_size >> 16) - 1;

#endif

    memset(s->font_offsets, '\0', sizeof(s->font_offsets));

    s->graphic_mode = -1; /* force full update */

    s->shift_control = 0;

    s->double_scan = 0;

    s->line_offset = 0;

    s->line_compare = 0;

    s->start_addr = 0;

    s->plane_updated = 0;

    s->last_cw = 0;

    s->last_ch = 0;

    s->last_width = 0;

    s->last_height = 0;

    s->last_scr_width = 0;

    s->last_scr_height = 0;

    s->cursor_start = 0;

    s->cursor_end = 0;

    s->cursor_offset = 0;

    memset(s->invalidated_y_table, '\0', sizeof(s->invalidated_y_table));

    memset(s->last_palette, '\0', sizeof(s->last_palette));

    memset(s->last_ch_attr, '\0', sizeof(s->last_ch_attr));

    switch (vga_retrace_method) {

    case VGA_RETRACE_DUMB:

        break;

    case VGA_RETRACE_PRECISE:

        memset(&s->retrace_info, 0, sizeof (s->retrace_info));

        break;

    }

}

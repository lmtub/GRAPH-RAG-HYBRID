static void sm501_disp_ctrl_write(void *opaque, hwaddr addr,

                                  uint64_t value, unsigned size)

{

    SM501State *s = (SM501State *)opaque;

    SM501_DPRINTF("sm501 disp ctrl regs : write addr=%x, val=%x\n",

                  (unsigned)addr, (unsigned)value);



    switch (addr) {

    case SM501_DC_PANEL_CONTROL:

        s->dc_panel_control = value & 0x0FFF73FF;

        break;

    case SM501_DC_PANEL_PANNING_CONTROL:

        s->dc_panel_panning_control = value & 0xFF3FFF3F;

        break;

    case SM501_DC_PANEL_FB_ADDR:

        s->dc_panel_fb_addr = value & 0x8FFFFFF0;

        break;

    case SM501_DC_PANEL_FB_OFFSET:

        s->dc_panel_fb_offset = value & 0x3FF03FF0;

        break;

    case SM501_DC_PANEL_FB_WIDTH:

        s->dc_panel_fb_width = value & 0x0FFF0FFF;

        break;

    case SM501_DC_PANEL_FB_HEIGHT:

        s->dc_panel_fb_height = value & 0x0FFF0FFF;

        break;

    case SM501_DC_PANEL_TL_LOC:

        s->dc_panel_tl_location = value & 0x07FF07FF;

        break;

    case SM501_DC_PANEL_BR_LOC:

        s->dc_panel_br_location = value & 0x07FF07FF;

        break;



    case SM501_DC_PANEL_H_TOT:

        s->dc_panel_h_total = value & 0x0FFF0FFF;

        break;

    case SM501_DC_PANEL_H_SYNC:

        s->dc_panel_h_sync = value & 0x00FF0FFF;

        break;

    case SM501_DC_PANEL_V_TOT:

        s->dc_panel_v_total = value & 0x0FFF0FFF;

        break;

    case SM501_DC_PANEL_V_SYNC:

        s->dc_panel_v_sync = value & 0x003F0FFF;

        break;



    case SM501_DC_PANEL_HWC_ADDR:

        s->dc_panel_hwc_addr = value & 0x8FFFFFF0;

        break;

    case SM501_DC_PANEL_HWC_LOC:

        s->dc_panel_hwc_location = value & 0x0FFF0FFF;

        break;

    case SM501_DC_PANEL_HWC_COLOR_1_2:

        s->dc_panel_hwc_color_1_2 = value;

        break;

    case SM501_DC_PANEL_HWC_COLOR_3:

        s->dc_panel_hwc_color_3 = value & 0x0000FFFF;

        break;



    case SM501_DC_CRT_CONTROL:

        s->dc_crt_control = value & 0x0003FFFF;

        break;

    case SM501_DC_CRT_FB_ADDR:

        s->dc_crt_fb_addr = value & 0x8FFFFFF0;

        break;

    case SM501_DC_CRT_FB_OFFSET:

        s->dc_crt_fb_offset = value & 0x3FF03FF0;

        break;

    case SM501_DC_CRT_H_TOT:

        s->dc_crt_h_total = value & 0x0FFF0FFF;

        break;

    case SM501_DC_CRT_H_SYNC:

        s->dc_crt_h_sync = value & 0x00FF0FFF;

        break;

    case SM501_DC_CRT_V_TOT:

        s->dc_crt_v_total = value & 0x0FFF0FFF;

        break;

    case SM501_DC_CRT_V_SYNC:

        s->dc_crt_v_sync = value & 0x003F0FFF;

        break;



    case SM501_DC_CRT_HWC_ADDR:

        s->dc_crt_hwc_addr = value & 0x8FFFFFF0;

        break;

    case SM501_DC_CRT_HWC_LOC:

        s->dc_crt_hwc_location = value & 0x0FFF0FFF;

        break;

    case SM501_DC_CRT_HWC_COLOR_1_2:

        s->dc_crt_hwc_color_1_2 = value;

        break;

    case SM501_DC_CRT_HWC_COLOR_3:

        s->dc_crt_hwc_color_3 = value & 0x0000FFFF;

        break;



    case SM501_DC_PANEL_PALETTE ... SM501_DC_PANEL_PALETTE + 0x400 * 3 - 4:

        sm501_palette_write(opaque, addr - SM501_DC_PANEL_PALETTE, value);

        break;



    default:

        printf("sm501 disp ctrl : not implemented register write."

               " addr=%x, val=%x\n", (int)addr, (unsigned)value);

        abort();

    }

}

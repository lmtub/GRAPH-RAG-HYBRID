static void exynos4210_fimd_write(void *opaque, target_phys_addr_t offset,

                              uint64_t val, unsigned size)

{

    Exynos4210fimdState *s = (Exynos4210fimdState *)opaque;

    unsigned w, i;

    uint32_t old_value;



    DPRINT_L2("write offset 0x%08x, value=%llu(0x%08llx)\n", offset,

            (long long unsigned int)val, (long long unsigned int)val);



    switch (offset) {

    case FIMD_VIDCON0:

        if ((val & FIMD_VIDCON0_ENVID_MASK) == FIMD_VIDCON0_ENVID_MASK) {

            exynos4210_fimd_enable(s, true);

        } else {

            if ((val & FIMD_VIDCON0_ENVID) == 0) {

                exynos4210_fimd_enable(s, false);

            }

        }

        s->vidcon[0] = val;

        break;

    case FIMD_VIDCON1:

        /* Leave read-only bits as is */

        val = (val & (~FIMD_VIDCON1_ROMASK)) |

                (s->vidcon[1] & FIMD_VIDCON1_ROMASK);

        s->vidcon[1] = val;

        break;

    case FIMD_VIDCON2 ... FIMD_VIDCON3:

        s->vidcon[(offset) >> 2] = val;

        break;

    case FIMD_VIDTCON_START ... FIMD_VIDTCON_END:

        s->vidtcon[(offset - FIMD_VIDTCON_START) >> 2] = val;

        break;

    case FIMD_WINCON_START ... FIMD_WINCON_END:

        w = (offset - FIMD_WINCON_START) >> 2;

        /* Window's current buffer ID */

        i = fimd_get_buffer_id(&s->window[w]);

        old_value = s->window[w].wincon;

        val = (val & ~FIMD_WINCON_ROMASK) |

                (s->window[w].wincon & FIMD_WINCON_ROMASK);

        if (w == 0) {

            /* Window 0 wincon ALPHA_MUL bit must always be 0 */

            val &= ~FIMD_WINCON_ALPHA_MUL;

        }

        exynos4210_fimd_trace_bppmode(s, w, val);

        switch (val & FIMD_WINCON_BUFSELECT) {

        case FIMD_WINCON_BUF0_SEL:

            val &= ~FIMD_WINCON_BUFSTATUS;

            break;

        case FIMD_WINCON_BUF1_SEL:

            val = (val & ~FIMD_WINCON_BUFSTAT_H) | FIMD_WINCON_BUFSTAT_L;

            break;

        case FIMD_WINCON_BUF2_SEL:

            if (val & FIMD_WINCON_BUFMODE) {

                val = (val & ~FIMD_WINCON_BUFSTAT_L) | FIMD_WINCON_BUFSTAT_H;

            }

            break;

        default:

            break;

        }

        s->window[w].wincon = val;

        exynos4210_fimd_update_win_bppmode(s, w);

        fimd_update_get_alpha(s, w);

        if ((i != fimd_get_buffer_id(&s->window[w])) ||

                (!(old_value & FIMD_WINCON_ENWIN) && (s->window[w].wincon &

                        FIMD_WINCON_ENWIN))) {

            fimd_update_memory_section(s, w);

        }

        break;

    case FIMD_SHADOWCON:

        old_value = s->shadowcon;

        s->shadowcon = val;

        for (w = 0; w < NUM_OF_WINDOWS; w++) {

            if (FIMD_WINDOW_PROTECTED(old_value, w) &&

                    !FIMD_WINDOW_PROTECTED(s->shadowcon, w)) {

                fimd_update_memory_section(s, w);

            }

        }

        break;

    case FIMD_WINCHMAP:

        s->winchmap = val;

        break;

    case FIMD_VIDOSD_START ... FIMD_VIDOSD_END:

        w = (offset - FIMD_VIDOSD_START) >> 4;

        i = ((offset - FIMD_VIDOSD_START) & 0xF) >> 2;

        switch (i) {

        case 0:

            old_value = s->window[w].lefttop_y;

            s->window[w].lefttop_x = (val >> FIMD_VIDOSD_HOR_SHIFT) &

                                      FIMD_VIDOSD_COORD_MASK;

            s->window[w].lefttop_y = (val >> FIMD_VIDOSD_VER_SHIFT) &

                                      FIMD_VIDOSD_COORD_MASK;

            if (s->window[w].lefttop_y != old_value) {

                fimd_update_memory_section(s, w);

            }

            break;

        case 1:

            old_value = s->window[w].rightbot_y;

            s->window[w].rightbot_x = (val >> FIMD_VIDOSD_HOR_SHIFT) &

                                       FIMD_VIDOSD_COORD_MASK;

            s->window[w].rightbot_y = (val >> FIMD_VIDOSD_VER_SHIFT) &

                                       FIMD_VIDOSD_COORD_MASK;

            if (s->window[w].rightbot_y != old_value) {

                fimd_update_memory_section(s, w);

            }

            break;

        case 2:

            if (w == 0) {

                s->window[w].osdsize = val;

            } else {

                s->window[w].alpha_val[0] =

                    unpack_upper_4((val & FIMD_VIDOSD_ALPHA_AEN0) >>

                    FIMD_VIDOSD_AEN0_SHIFT) |

                    (s->window[w].alpha_val[0] & FIMD_VIDALPHA_ALPHA_LOWER);

                s->window[w].alpha_val[1] =

                    unpack_upper_4(val & FIMD_VIDOSD_ALPHA_AEN1) |

                    (s->window[w].alpha_val[1] & FIMD_VIDALPHA_ALPHA_LOWER);

            }

            break;

        case 3:

            if (w != 1 && w != 2) {

                DPRINT_ERROR("Bad write offset 0x%08x\n", offset);

                return;

            }

            s->window[w].osdsize = val;

            break;

        }

        break;

    case FIMD_VIDWADD0_START ... FIMD_VIDWADD0_END:

        w = (offset - FIMD_VIDWADD0_START) >> 3;

        i = ((offset - FIMD_VIDWADD0_START) >> 2) & 1;

        if (i == fimd_get_buffer_id(&s->window[w]) &&

                s->window[w].buf_start[i] != val) {

            s->window[w].buf_start[i] = val;

            fimd_update_memory_section(s, w);

            break;

        }

        s->window[w].buf_start[i] = val;

        break;

    case FIMD_VIDWADD1_START ... FIMD_VIDWADD1_END:

        w = (offset - FIMD_VIDWADD1_START) >> 3;

        i = ((offset - FIMD_VIDWADD1_START) >> 2) & 1;

        s->window[w].buf_end[i] = val;

        break;

    case FIMD_VIDWADD2_START ... FIMD_VIDWADD2_END:

        w = (offset - FIMD_VIDWADD2_START) >> 2;

        if (((val & FIMD_VIDWADD2_PAGEWIDTH) != s->window[w].virtpage_width) ||

            (((val >> FIMD_VIDWADD2_OFFSIZE_SHIFT) & FIMD_VIDWADD2_OFFSIZE) !=

                        s->window[w].virtpage_offsize)) {

            s->window[w].virtpage_width = val & FIMD_VIDWADD2_PAGEWIDTH;

            s->window[w].virtpage_offsize =

                (val >> FIMD_VIDWADD2_OFFSIZE_SHIFT) & FIMD_VIDWADD2_OFFSIZE;

            fimd_update_memory_section(s, w);

        }

        break;

    case FIMD_VIDINTCON0:

        s->vidintcon[0] = val;

        break;

    case FIMD_VIDINTCON1:

        s->vidintcon[1] &= ~(val & 7);

        exynos4210_fimd_update_irq(s);

        break;

    case FIMD_WKEYCON_START ... FIMD_WKEYCON_END:

        w = ((offset - FIMD_WKEYCON_START) >> 3) + 1;

        i = ((offset - FIMD_WKEYCON_START) >> 2) & 1;

        s->window[w].keycon[i] = val;

        break;

    case FIMD_WKEYALPHA_START ... FIMD_WKEYALPHA_END:

        w = ((offset - FIMD_WKEYALPHA_START) >> 2) + 1;

        s->window[w].keyalpha = val;

        break;

    case FIMD_DITHMODE:

        s->dithmode = val;

        break;

    case FIMD_WINMAP_START ... FIMD_WINMAP_END:

        w = (offset - FIMD_WINMAP_START) >> 2;

        old_value = s->window[w].winmap;

        s->window[w].winmap = val;

        if ((val & FIMD_WINMAP_EN) ^ (old_value & FIMD_WINMAP_EN)) {

            exynos4210_fimd_invalidate(s);

            exynos4210_fimd_update_win_bppmode(s, w);

            exynos4210_fimd_trace_bppmode(s, w, 0xFFFFFFFF);

            exynos4210_fimd_update(s);

        }

        break;

    case FIMD_WPALCON_HIGH ... FIMD_WPALCON_LOW:

        i = (offset - FIMD_WPALCON_HIGH) >> 2;

        s->wpalcon[i] = val;

        if (s->wpalcon[1] & FIMD_WPALCON_UPDATEEN) {

            for (w = 0; w < NUM_OF_WINDOWS; w++) {

                exynos4210_fimd_update_win_bppmode(s, w);

                fimd_update_get_alpha(s, w);

            }

        }

        break;

    case FIMD_TRIGCON:

        val = (val & ~FIMD_TRIGCON_ROMASK) | (s->trigcon & FIMD_TRIGCON_ROMASK);

        s->trigcon = val;

        break;

    case FIMD_I80IFCON_START ... FIMD_I80IFCON_END:

        s->i80ifcon[(offset - FIMD_I80IFCON_START) >> 2] = val;

        break;

    case FIMD_COLORGAINCON:

        s->colorgaincon = val;

        break;

    case FIMD_LDI_CMDCON0 ... FIMD_LDI_CMDCON1:

        s->ldi_cmdcon[(offset - FIMD_LDI_CMDCON0) >> 2] = val;

        break;

    case FIMD_SIFCCON0 ... FIMD_SIFCCON2:

        i = (offset - FIMD_SIFCCON0) >> 2;

        if (i != 2) {

            s->sifccon[i] = val;

        }

        break;

    case FIMD_HUECOEFCR_START ... FIMD_HUECOEFCR_END:

        i = (offset - FIMD_HUECOEFCR_START) >> 2;

        s->huecoef_cr[i] = val;

        break;

    case FIMD_HUECOEFCB_START ... FIMD_HUECOEFCB_END:

        i = (offset - FIMD_HUECOEFCB_START) >> 2;

        s->huecoef_cb[i] = val;

        break;

    case FIMD_HUEOFFSET:

        s->hueoffset = val;

        break;

    case FIMD_VIDWALPHA_START ... FIMD_VIDWALPHA_END:

        w = ((offset - FIMD_VIDWALPHA_START) >> 3);

        i = ((offset - FIMD_VIDWALPHA_START) >> 2) & 1;

        if (w == 0) {

            s->window[w].alpha_val[i] = val;

        } else {

            s->window[w].alpha_val[i] = (val & FIMD_VIDALPHA_ALPHA_LOWER) |

                (s->window[w].alpha_val[i] & FIMD_VIDALPHA_ALPHA_UPPER);

        }

        break;

    case FIMD_BLENDEQ_START ... FIMD_BLENDEQ_END:

        s->window[(offset - FIMD_BLENDEQ_START) >> 2].blendeq = val;

        break;

    case FIMD_BLENDCON:

        old_value = s->blendcon;

        s->blendcon = val;

        if ((s->blendcon & FIMD_ALPHA_8BIT) != (old_value & FIMD_ALPHA_8BIT)) {

            for (w = 0; w < NUM_OF_WINDOWS; w++) {

                fimd_update_get_alpha(s, w);

            }

        }

        break;

    case FIMD_WRTQOSCON_START ... FIMD_WRTQOSCON_END:

        s->window[(offset - FIMD_WRTQOSCON_START) >> 2].rtqoscon = val;

        break;

    case FIMD_I80IFCMD_START ... FIMD_I80IFCMD_END:

        s->i80ifcmd[(offset - FIMD_I80IFCMD_START) >> 2] = val;

        break;

    case FIMD_VIDW0ADD0_B2 ... FIMD_VIDW4ADD0_B2:

        if (offset & 0x0004) {

            DPRINT_ERROR("bad write offset 0x%08x\n", offset);

            break;

        }

        w = (offset - FIMD_VIDW0ADD0_B2) >> 3;

        if (fimd_get_buffer_id(&s->window[w]) == 2 &&

                s->window[w].buf_start[2] != val) {

            s->window[w].buf_start[2] = val;

            fimd_update_memory_section(s, w);

            break;

        }

        s->window[w].buf_start[2] = val;

        break;

    case FIMD_SHD_ADD0_START ... FIMD_SHD_ADD0_END:

        if (offset & 0x0004) {

            DPRINT_ERROR("bad write offset 0x%08x\n", offset);

            break;

        }

        s->window[(offset - FIMD_SHD_ADD0_START) >> 3].shadow_buf_start = val;

        break;

    case FIMD_SHD_ADD1_START ... FIMD_SHD_ADD1_END:

        if (offset & 0x0004) {

            DPRINT_ERROR("bad write offset 0x%08x\n", offset);

            break;

        }

        s->window[(offset - FIMD_SHD_ADD1_START) >> 3].shadow_buf_end = val;

        break;

    case FIMD_SHD_ADD2_START ... FIMD_SHD_ADD2_END:

        s->window[(offset - FIMD_SHD_ADD2_START) >> 2].shadow_buf_size = val;

        break;

    case FIMD_PAL_MEM_START ... FIMD_PAL_MEM_END:

        w = (offset - FIMD_PAL_MEM_START) >> 10;

        i = ((offset - FIMD_PAL_MEM_START) >> 2) & 0xFF;

        s->window[w].palette[i] = val;

        break;

    case FIMD_PALMEM_AL_START ... FIMD_PALMEM_AL_END:

        /* Palette memory aliases for windows 0 and 1 */

        w = (offset - FIMD_PALMEM_AL_START) >> 10;

        i = ((offset - FIMD_PALMEM_AL_START) >> 2) & 0xFF;

        s->window[w].palette[i] = val;

        break;

    default:

        DPRINT_ERROR("bad write offset 0x%08x\n", offset);

        break;

    }

}

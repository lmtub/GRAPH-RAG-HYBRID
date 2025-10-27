static uint64_t exynos4210_fimd_read(void *opaque, target_phys_addr_t offset,

                                  unsigned size)

{

    Exynos4210fimdState *s = (Exynos4210fimdState *)opaque;

    int w, i;

    uint32_t ret = 0;



    DPRINT_L2("read offset 0x%08x\n", offset);



    switch (offset) {

    case FIMD_VIDCON0 ... FIMD_VIDCON3:

        return s->vidcon[(offset - FIMD_VIDCON0) >> 2];

    case FIMD_VIDTCON_START ... FIMD_VIDTCON_END:

        return s->vidtcon[(offset - FIMD_VIDTCON_START) >> 2];

    case FIMD_WINCON_START ... FIMD_WINCON_END:

        return s->window[(offset - FIMD_WINCON_START) >> 2].wincon;

    case FIMD_SHADOWCON:

        return s->shadowcon;

    case FIMD_WINCHMAP:

        return s->winchmap;

    case FIMD_VIDOSD_START ... FIMD_VIDOSD_END:

        w = (offset - FIMD_VIDOSD_START) >> 4;

        i = ((offset - FIMD_VIDOSD_START) & 0xF) >> 2;

        switch (i) {

        case 0:

            ret = ((s->window[w].lefttop_x & FIMD_VIDOSD_COORD_MASK) <<

            FIMD_VIDOSD_HOR_SHIFT) |

            (s->window[w].lefttop_y & FIMD_VIDOSD_COORD_MASK);

            break;

        case 1:

            ret = ((s->window[w].rightbot_x & FIMD_VIDOSD_COORD_MASK) <<

                FIMD_VIDOSD_HOR_SHIFT) |

                (s->window[w].rightbot_y & FIMD_VIDOSD_COORD_MASK);

            break;

        case 2:

            if (w == 0) {

                ret = s->window[w].osdsize;

            } else {

                ret = (pack_upper_4(s->window[w].alpha_val[0]) <<

                    FIMD_VIDOSD_AEN0_SHIFT) |

                    pack_upper_4(s->window[w].alpha_val[1]);

            }

            break;

        case 3:

            if (w != 1 && w != 2) {

                DPRINT_ERROR("bad read offset 0x%08x\n", offset);

                return 0xBAADBAAD;

            }

            ret = s->window[w].osdsize;

            break;

        }

        return ret;

    case FIMD_VIDWADD0_START ... FIMD_VIDWADD0_END:

        w = (offset - FIMD_VIDWADD0_START) >> 3;

        i = ((offset - FIMD_VIDWADD0_START) >> 2) & 1;

        return s->window[w].buf_start[i];

    case FIMD_VIDWADD1_START ... FIMD_VIDWADD1_END:

        w = (offset - FIMD_VIDWADD1_START) >> 3;

        i = ((offset - FIMD_VIDWADD1_START) >> 2) & 1;

        return s->window[w].buf_end[i];

    case FIMD_VIDWADD2_START ... FIMD_VIDWADD2_END:

        w = (offset - FIMD_VIDWADD2_START) >> 2;

        return s->window[w].virtpage_width | (s->window[w].virtpage_offsize <<

            FIMD_VIDWADD2_OFFSIZE_SHIFT);

    case FIMD_VIDINTCON0 ... FIMD_VIDINTCON1:

        return s->vidintcon[(offset - FIMD_VIDINTCON0) >> 2];

    case FIMD_WKEYCON_START ... FIMD_WKEYCON_END:

        w = ((offset - FIMD_WKEYCON_START) >> 3) + 1;

        i = ((offset - FIMD_WKEYCON_START) >> 2) & 1;

        return s->window[w].keycon[i];

    case FIMD_WKEYALPHA_START ... FIMD_WKEYALPHA_END:

        w = ((offset - FIMD_WKEYALPHA_START) >> 2) + 1;

        return s->window[w].keyalpha;

    case FIMD_DITHMODE:

        return s->dithmode;

    case FIMD_WINMAP_START ... FIMD_WINMAP_END:

        return s->window[(offset - FIMD_WINMAP_START) >> 2].winmap;

    case FIMD_WPALCON_HIGH ... FIMD_WPALCON_LOW:

        return s->wpalcon[(offset - FIMD_WPALCON_HIGH) >> 2];

    case FIMD_TRIGCON:

        return s->trigcon;

    case FIMD_I80IFCON_START ... FIMD_I80IFCON_END:

        return s->i80ifcon[(offset - FIMD_I80IFCON_START) >> 2];

    case FIMD_COLORGAINCON:

        return s->colorgaincon;

    case FIMD_LDI_CMDCON0 ... FIMD_LDI_CMDCON1:

        return s->ldi_cmdcon[(offset - FIMD_LDI_CMDCON0) >> 2];

    case FIMD_SIFCCON0 ... FIMD_SIFCCON2:

        i = (offset - FIMD_SIFCCON0) >> 2;

        return s->sifccon[i];

    case FIMD_HUECOEFCR_START ... FIMD_HUECOEFCR_END:

        i = (offset - FIMD_HUECOEFCR_START) >> 2;

        return s->huecoef_cr[i];

    case FIMD_HUECOEFCB_START ... FIMD_HUECOEFCB_END:

        i = (offset - FIMD_HUECOEFCB_START) >> 2;

        return s->huecoef_cb[i];

    case FIMD_HUEOFFSET:

        return s->hueoffset;

    case FIMD_VIDWALPHA_START ... FIMD_VIDWALPHA_END:

        w = ((offset - FIMD_VIDWALPHA_START) >> 3);

        i = ((offset - FIMD_VIDWALPHA_START) >> 2) & 1;

        return s->window[w].alpha_val[i] &

                (w == 0 ? 0xFFFFFF : FIMD_VIDALPHA_ALPHA_LOWER);

    case FIMD_BLENDEQ_START ... FIMD_BLENDEQ_END:

        return s->window[(offset - FIMD_BLENDEQ_START) >> 2].blendeq;

    case FIMD_BLENDCON:

        return s->blendcon;

    case FIMD_WRTQOSCON_START ... FIMD_WRTQOSCON_END:

        return s->window[(offset - FIMD_WRTQOSCON_START) >> 2].rtqoscon;

    case FIMD_I80IFCMD_START ... FIMD_I80IFCMD_END:

        return s->i80ifcmd[(offset - FIMD_I80IFCMD_START) >> 2];

    case FIMD_VIDW0ADD0_B2 ... FIMD_VIDW4ADD0_B2:

        if (offset & 0x0004) {

            break;

        }

        return s->window[(offset - FIMD_VIDW0ADD0_B2) >> 3].buf_start[2];

    case FIMD_SHD_ADD0_START ... FIMD_SHD_ADD0_END:

        if (offset & 0x0004) {

            break;

        }

        return s->window[(offset - FIMD_SHD_ADD0_START) >> 3].shadow_buf_start;

    case FIMD_SHD_ADD1_START ... FIMD_SHD_ADD1_END:

        if (offset & 0x0004) {

            break;

        }

        return s->window[(offset - FIMD_SHD_ADD1_START) >> 3].shadow_buf_end;

    case FIMD_SHD_ADD2_START ... FIMD_SHD_ADD2_END:

        return s->window[(offset - FIMD_SHD_ADD2_START) >> 2].shadow_buf_size;

    case FIMD_PAL_MEM_START ... FIMD_PAL_MEM_END:

        w = (offset - FIMD_PAL_MEM_START) >> 10;

        i = ((offset - FIMD_PAL_MEM_START) >> 2) & 0xFF;

        return s->window[w].palette[i];

    case FIMD_PALMEM_AL_START ... FIMD_PALMEM_AL_END:

        /* Palette aliases for win 0,1 */

        w = (offset - FIMD_PALMEM_AL_START) >> 10;

        i = ((offset - FIMD_PALMEM_AL_START) >> 2) & 0xFF;

        return s->window[w].palette[i];

    }



    DPRINT_ERROR("bad read offset 0x%08x\n", offset);

    return 0xBAADBAAD;

}

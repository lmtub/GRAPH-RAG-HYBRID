static int wm8750_tx(I2CSlave *i2c, uint8_t data)

{

    WM8750State *s = (WM8750State *) i2c;

    uint8_t cmd;

    uint16_t value;



    if (s->i2c_len >= 2) {

        printf("%s: long message (%i bytes)\n", __FUNCTION__, s->i2c_len);

#ifdef VERBOSE

        return 1;

#endif

    }

    s->i2c_data[s->i2c_len ++] = data;

    if (s->i2c_len != 2)

        return 0;



    cmd = s->i2c_data[0] >> 1;

    value = ((s->i2c_data[0] << 8) | s->i2c_data[1]) & 0x1ff;



    switch (cmd) {

    case WM8750_LADCIN:	/* ADC Signal Path Control (Left) */

        s->diff[0] = (((value >> 6) & 3) == 3);	/* LINSEL */

        if (s->diff[0])

            s->in[0] = &s->adc_voice[0 + s->ds * 1];

        else

            s->in[0] = &s->adc_voice[((value >> 6) & 3) * 1 + 0];

        break;



    case WM8750_RADCIN:	/* ADC Signal Path Control (Right) */

        s->diff[1] = (((value >> 6) & 3) == 3);	/* RINSEL */

        if (s->diff[1])

            s->in[1] = &s->adc_voice[0 + s->ds * 1];

        else

            s->in[1] = &s->adc_voice[((value >> 6) & 3) * 1 + 0];

        break;



    case WM8750_ADCIN:	/* ADC Input Mode */

        s->ds = (value >> 8) & 1;	/* DS */

        if (s->diff[0])

            s->in[0] = &s->adc_voice[0 + s->ds * 1];

        if (s->diff[1])

            s->in[1] = &s->adc_voice[0 + s->ds * 1];

        s->monomix[0] = (value >> 6) & 3;	/* MONOMIX */

        break;



    case WM8750_ADCTL1:	/* Additional Control (1) */

        s->monomix[1] = (value >> 1) & 1;	/* DMONOMIX */

        break;



    case WM8750_PWR1:	/* Power Management (1) */

        s->enable = ((value >> 6) & 7) == 3;	/* VMIDSEL, VREF */

        wm8750_set_format(s);

        break;



    case WM8750_LINVOL:	/* Left Channel PGA */

        s->invol[0] = value & 0x3f;		/* LINVOL */

        s->inmute[0] = (value >> 7) & 1;	/* LINMUTE */

        wm8750_vol_update(s);

        break;



    case WM8750_RINVOL:	/* Right Channel PGA */

        s->invol[1] = value & 0x3f;		/* RINVOL */

        s->inmute[1] = (value >> 7) & 1;	/* RINMUTE */

        wm8750_vol_update(s);

        break;



    case WM8750_ADCDAC:	/* ADC and DAC Control */

        s->pol = (value >> 5) & 3;		/* ADCPOL */

        s->mute = (value >> 3) & 1;		/* DACMU */

        wm8750_vol_update(s);

        break;



    case WM8750_ADCTL3:	/* Additional Control (3) */

        break;



    case WM8750_LADC:	/* Left ADC Digital Volume */

        s->invol[2] = value & 0xff;		/* LADCVOL */

        wm8750_vol_update(s);

        break;



    case WM8750_RADC:	/* Right ADC Digital Volume */

        s->invol[3] = value & 0xff;		/* RADCVOL */

        wm8750_vol_update(s);

        break;



    case WM8750_ALC1:	/* ALC Control (1) */

        s->alc = (value >> 7) & 3;		/* ALCSEL */

        break;



    case WM8750_NGATE:	/* Noise Gate Control */

    case WM8750_3D:	/* 3D enhance */

        break;



    case WM8750_LDAC:	/* Left Channel Digital Volume */

        s->outvol[0] = value & 0xff;		/* LDACVOL */

        wm8750_vol_update(s);

        break;



    case WM8750_RDAC:	/* Right Channel Digital Volume */

        s->outvol[1] = value & 0xff;		/* RDACVOL */

        wm8750_vol_update(s);

        break;



    case WM8750_BASS:	/* Bass Control */

        break;



    case WM8750_LOUTM1:	/* Left Mixer Control (1) */

        s->path[0] = (value >> 8) & 1;		/* LD2LO */

        /* TODO: mute/unmute respective paths */

        wm8750_vol_update(s);

        break;



    case WM8750_LOUTM2:	/* Left Mixer Control (2) */

        s->path[1] = (value >> 8) & 1;		/* RD2LO */

        /* TODO: mute/unmute respective paths */

        wm8750_vol_update(s);

        break;



    case WM8750_ROUTM1:	/* Right Mixer Control (1) */

        s->path[2] = (value >> 8) & 1;		/* LD2RO */

        /* TODO: mute/unmute respective paths */

        wm8750_vol_update(s);

        break;



    case WM8750_ROUTM2:	/* Right Mixer Control (2) */

        s->path[3] = (value >> 8) & 1;		/* RD2RO */

        /* TODO: mute/unmute respective paths */

        wm8750_vol_update(s);

        break;



    case WM8750_MOUTM1:	/* Mono Mixer Control (1) */

        s->mpath[0] = (value >> 8) & 1;		/* LD2MO */

        /* TODO: mute/unmute respective paths */

        wm8750_vol_update(s);

        break;



    case WM8750_MOUTM2:	/* Mono Mixer Control (2) */

        s->mpath[1] = (value >> 8) & 1;		/* RD2MO */

        /* TODO: mute/unmute respective paths */

        wm8750_vol_update(s);

        break;



    case WM8750_LOUT1V:	/* LOUT1 Volume */

        s->outvol[2] = value & 0x7f;		/* LOUT1VOL */

        wm8750_vol_update(s);

        break;



    case WM8750_LOUT2V:	/* LOUT2 Volume */

        s->outvol[4] = value & 0x7f;		/* LOUT2VOL */

        wm8750_vol_update(s);

        break;



    case WM8750_ROUT1V:	/* ROUT1 Volume */

        s->outvol[3] = value & 0x7f;		/* ROUT1VOL */

        wm8750_vol_update(s);

        break;



    case WM8750_ROUT2V:	/* ROUT2 Volume */

        s->outvol[5] = value & 0x7f;		/* ROUT2VOL */

        wm8750_vol_update(s);

        break;



    case WM8750_MOUTV:	/* MONOOUT Volume */

        s->outvol[6] = value & 0x7f;		/* MONOOUTVOL */

        wm8750_vol_update(s);

        break;



    case WM8750_ADCTL2:	/* Additional Control (2) */

        break;



    case WM8750_PWR2:	/* Power Management (2) */

        s->power = value & 0x7e;

        /* TODO: mute/unmute respective paths */

        wm8750_vol_update(s);

        break;



    case WM8750_IFACE:	/* Digital Audio Interface Format */

        s->format = value;

        s->master = (value >> 6) & 1;			/* MS */

        wm8750_clk_update(s, s->master);

        break;



    case WM8750_SRATE:	/* Clocking and Sample Rate Control */

        s->rate = &wm_rate_table[(value >> 1) & 0x1f];

        wm8750_clk_update(s, 0);

        break;



    case WM8750_RESET:	/* Reset */

        wm8750_reset(&s->i2c);

        break;



#ifdef VERBOSE

    default:

        printf("%s: unknown register %02x\n", __FUNCTION__, cmd);

#endif

    }



    return 0;

}

void lm4549_write(lm4549_state *s,

                  target_phys_addr_t offset, uint32_t value)

{

    uint16_t *regfile = s->regfile;



    assert(offset < 128);

    DPRINTF("write [0x%02x] = 0x%04x\n", offset, value);



    switch (offset) {

    case LM4549_Reset:

        lm4549_reset(s);

        break;



    case LM4549_PCM_Front_DAC_Rate:

        regfile[LM4549_PCM_Front_DAC_Rate] = value;

        DPRINTF("DAC rate change = %i\n", value);



        /* Re-open a voice with the new sample rate */

        struct audsettings as;

        as.freq = value;

        as.nchannels = 2;

        as.fmt = AUD_FMT_S16;

        as.endianness = 0;



        s->voice = AUD_open_out(

            &s->card,

            s->voice,

            "lm4549.out",

            s,

            lm4549_audio_out_callback,

            &as

        );

        break;



    case LM4549_Powerdown_Ctrl_Stat:

        value &= ~0xf;

        value |= regfile[LM4549_Powerdown_Ctrl_Stat] & 0xf;

        regfile[LM4549_Powerdown_Ctrl_Stat] = value;

        break;



    case LM4549_Ext_Audio_ID:

    case LM4549_Vendor_ID1:

    case LM4549_Vendor_ID2:

        DPRINTF("Write to read-only register 0x%x\n", (int)offset);

        break;



    default:

        /* Store the new value */

        regfile[offset] = value;

        break;

    }

}

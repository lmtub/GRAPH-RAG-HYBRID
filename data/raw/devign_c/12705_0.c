static void nam_writew (void *opaque, uint32_t addr, uint32_t val)

{

    PCIAC97LinkState *d = opaque;

    AC97LinkState *s = &d->ac97;

    uint32_t index = addr - s->base[0];

    s->cas = 0;

    switch (index) {

    case AC97_Reset:

        mixer_reset (s);

        break;

    case AC97_Powerdown_Ctrl_Stat:

        val &= ~0xf;

        val |= mixer_load (s, index) & 0xf;

        mixer_store (s, index, val);

        break;

#ifdef USE_MIXER

    case AC97_Master_Volume_Mute:

        set_volume (s, index, AUD_MIXER_VOLUME, val);

        break;

    case AC97_PCM_Out_Volume_Mute:

        set_volume (s, index, AUD_MIXER_PCM, val);

        break;

    case AC97_Line_In_Volume_Mute:

        set_volume (s, index, AUD_MIXER_LINE_IN, val);

        break;

    case AC97_Record_Select:

        record_select (s, val);

        break;

#endif

    case AC97_Vendor_ID1:

    case AC97_Vendor_ID2:

        dolog ("Attempt to write vendor ID to %#x\n", val);

        break;

    case AC97_Extended_Audio_ID:

        dolog ("Attempt to write extended audio ID to %#x\n", val);

        break;

    case AC97_Extended_Audio_Ctrl_Stat:

        if (!(val & EACS_VRA)) {

            mixer_store (s, AC97_PCM_Front_DAC_Rate, 0xbb80);

            mixer_store (s, AC97_PCM_LR_ADC_Rate,    0xbb80);

            open_voice (s, PI_INDEX, 48000);

            open_voice (s, PO_INDEX, 48000);

        }

        if (!(val & EACS_VRM)) {

            mixer_store (s, AC97_MIC_ADC_Rate, 0xbb80);

            open_voice (s, MC_INDEX, 48000);

        }

        dolog ("Setting extended audio control to %#x\n", val);

        mixer_store (s, AC97_Extended_Audio_Ctrl_Stat, val);

        break;

    case AC97_PCM_Front_DAC_Rate:

        if (mixer_load (s, AC97_Extended_Audio_Ctrl_Stat) & EACS_VRA) {

            mixer_store (s, index, val);

            dolog ("Set front DAC rate to %d\n", val);

            open_voice (s, PO_INDEX, val);

        }

        else {

            dolog ("Attempt to set front DAC rate to %d, "

                   "but VRA is not set\n",

                   val);

        }

        break;

    case AC97_MIC_ADC_Rate:

        if (mixer_load (s, AC97_Extended_Audio_Ctrl_Stat) & EACS_VRM) {

            mixer_store (s, index, val);

            dolog ("Set MIC ADC rate to %d\n", val);

            open_voice (s, MC_INDEX, val);

        }

        else {

            dolog ("Attempt to set MIC ADC rate to %d, "

                   "but VRM is not set\n",

                   val);

        }

        break;

    case AC97_PCM_LR_ADC_Rate:

        if (mixer_load (s, AC97_Extended_Audio_Ctrl_Stat) & EACS_VRA) {

            mixer_store (s, index, val);

            dolog ("Set front LR ADC rate to %d\n", val);

            open_voice (s, PI_INDEX, val);

        }

        else {

            dolog ("Attempt to set LR ADC rate to %d, but VRA is not set\n",

                    val);

        }

        break;

    default:

        dolog ("U nam writew %#x <- %#x\n", addr, val);

        mixer_store (s, index, val);

        break;

    }

}

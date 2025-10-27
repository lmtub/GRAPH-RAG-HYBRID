static void kbd_write_command(void *opaque, hwaddr addr,

                              uint64_t val, unsigned size)

{

    KBDState *s = opaque;



    DPRINTF("kbd: write cmd=0x%02" PRIx64 "\n", val);



    /* Bits 3-0 of the output port P2 of the keyboard controller may be pulsed

     * low for approximately 6 micro seconds. Bits 3-0 of the KBD_CCMD_PULSE

     * command specify the output port bits to be pulsed.

     * 0: Bit should be pulsed. 1: Bit should not be modified.

     * The only useful version of this command is pulsing bit 0,

     * which does a CPU reset.

     */

    if((val & KBD_CCMD_PULSE_BITS_3_0) == KBD_CCMD_PULSE_BITS_3_0) {

        if(!(val & 1))

            val = KBD_CCMD_RESET;

        else

            val = KBD_CCMD_NO_OP;

    }



    switch(val) {

    case KBD_CCMD_READ_MODE:

        kbd_queue(s, s->mode, 0);

        break;

    case KBD_CCMD_WRITE_MODE:

    case KBD_CCMD_WRITE_OBUF:

    case KBD_CCMD_WRITE_AUX_OBUF:

    case KBD_CCMD_WRITE_MOUSE:

    case KBD_CCMD_WRITE_OUTPORT:

        s->write_cmd = val;

        break;

    case KBD_CCMD_MOUSE_DISABLE:

        s->mode |= KBD_MODE_DISABLE_MOUSE;

        break;

    case KBD_CCMD_MOUSE_ENABLE:

        s->mode &= ~KBD_MODE_DISABLE_MOUSE;

        break;

    case KBD_CCMD_TEST_MOUSE:

        kbd_queue(s, 0x00, 0);

        break;

    case KBD_CCMD_SELF_TEST:

        s->status |= KBD_STAT_SELFTEST;

        kbd_queue(s, 0x55, 0);

        break;

    case KBD_CCMD_KBD_TEST:

        kbd_queue(s, 0x00, 0);

        break;

    case KBD_CCMD_KBD_DISABLE:

        s->mode |= KBD_MODE_DISABLE_KBD;

        kbd_update_irq(s);

        break;

    case KBD_CCMD_KBD_ENABLE:

        s->mode &= ~KBD_MODE_DISABLE_KBD;

        kbd_update_irq(s);

        break;

    case KBD_CCMD_READ_INPORT:

        kbd_queue(s, 0x80, 0);

        break;

    case KBD_CCMD_READ_OUTPORT:

        kbd_queue(s, s->outport, 0);

        break;

    case KBD_CCMD_ENABLE_A20:

        if (s->a20_out) {

            qemu_irq_raise(*s->a20_out);

        }

        s->outport |= KBD_OUT_A20;

        break;

    case KBD_CCMD_DISABLE_A20:

        if (s->a20_out) {

            qemu_irq_lower(*s->a20_out);

        }

        s->outport &= ~KBD_OUT_A20;

        break;

    case KBD_CCMD_RESET:

        qemu_system_reset_request();

        break;

    case KBD_CCMD_NO_OP:

        /* ignore that */

        break;

    default:

        fprintf(stderr, "qemu: unsupported keyboard cmd=0x%02x\n", (int)val);

        break;

    }

}

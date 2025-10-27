static void vgafb_write(void *opaque, target_phys_addr_t addr, uint64_t value,

                        unsigned size)

{

    MilkymistVgafbState *s = opaque;



    trace_milkymist_vgafb_memory_write(addr, value);



    addr >>= 2;

    switch (addr) {

    case R_CTRL:

        s->regs[addr] = value;

        vgafb_resize(s);

        break;

    case R_HSYNC_START:

    case R_HSYNC_END:

    case R_HSCAN:

    case R_VSYNC_START:

    case R_VSYNC_END:

    case R_VSCAN:

    case R_BURST_COUNT:

    case R_DDC:

    case R_SOURCE_CLOCK:

        s->regs[addr] = value;

        break;

    case R_BASEADDRESS:

        if (value & 0x1f) {

            error_report("milkymist_vgafb: framebuffer base address have to "

                     "be 32 byte aligned");

            break;

        }

        s->regs[addr] = value & s->fb_mask;

        s->invalidate = 1;

        break;

    case R_HRES:

    case R_VRES:

        s->regs[addr] = value;

        vgafb_resize(s);

        break;

    case R_BASEADDRESS_ACT:

        error_report("milkymist_vgafb: write to read-only register 0x"

                TARGET_FMT_plx, addr << 2);

        break;



    default:

        error_report("milkymist_vgafb: write access to unknown register 0x"

                TARGET_FMT_plx, addr << 2);

        break;

    }

}

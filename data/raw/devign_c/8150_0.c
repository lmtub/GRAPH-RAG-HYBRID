int cpu_x86_signal_handler(int host_signum, struct siginfo *info, 

                           void *puc)

{

    uint32_t *regs = (uint32_t *)(info + 1);

    void *sigmask = (regs + 20);

    unsigned long pc;

    int is_write;

    uint32_t insn;

    

    /* XXX: is there a standard glibc define ? */

    pc = regs[1];

    /* XXX: need kernel patch to get write flag faster */

    is_write = 0;

    insn = *(uint32_t *)pc;

    if ((insn >> 30) == 3) {

      switch((insn >> 19) & 0x3f) {

      case 0x05: // stb

      case 0x06: // sth

      case 0x04: // st

      case 0x07: // std

      case 0x24: // stf

      case 0x27: // stdf

      case 0x25: // stfsr

	is_write = 1;

	break;

      }

    }

    return handle_cpu_signal(pc, (unsigned long)info->si_addr, 

                             is_write, sigmask);

}

static uint32_t nabm_readb (void *opaque, uint32_t addr)

{

    PCIAC97LinkState *d = opaque;

    AC97LinkState *s = &d->ac97;

    AC97BusMasterRegs *r = NULL;

    uint32_t index = addr - s->base[1];

    uint32_t val = ~0U;



    switch (index) {

    case CAS:

        dolog ("CAS %d\n", s->cas);

        val = s->cas;

        s->cas = 1;

        break;

    case PI_CIV:

    case PO_CIV:

    case MC_CIV:

        r = &s->bm_regs[GET_BM (index)];

        val = r->civ;

        dolog ("CIV[%d] -> %#x\n", GET_BM (index), val);

        break;

    case PI_LVI:

    case PO_LVI:

    case MC_LVI:

        r = &s->bm_regs[GET_BM (index)];

        val = r->lvi;

        dolog ("LVI[%d] -> %#x\n", GET_BM (index), val);

        break;

    case PI_PIV:

    case PO_PIV:

    case MC_PIV:

        r = &s->bm_regs[GET_BM (index)];

        val = r->piv;

        dolog ("PIV[%d] -> %#x\n", GET_BM (index), val);

        break;

    case PI_CR:

    case PO_CR:

    case MC_CR:

        r = &s->bm_regs[GET_BM (index)];

        val = r->cr;

        dolog ("CR[%d] -> %#x\n", GET_BM (index), val);

        break;

    case PI_SR:

    case PO_SR:

    case MC_SR:

        r = &s->bm_regs[GET_BM (index)];

        val = r->sr & 0xff;

        dolog ("SRb[%d] -> %#x\n", GET_BM (index), val);

        break;

    default:

        dolog ("U nabm readb %#x -> %#x\n", addr, val);

        break;

    }

    return val;

}

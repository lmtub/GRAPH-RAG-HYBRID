static uint32_t nabm_readl (void *opaque, uint32_t addr)

{

    PCIAC97LinkState *d = opaque;

    AC97LinkState *s = &d->ac97;

    AC97BusMasterRegs *r = NULL;

    uint32_t index = addr - s->base[1];

    uint32_t val = ~0U;



    switch (index) {

    case PI_BDBAR:

    case PO_BDBAR:

    case MC_BDBAR:

        r = &s->bm_regs[GET_BM (index)];

        val = r->bdbar;

        dolog ("BMADDR[%d] -> %#x\n", GET_BM (index), val);

        break;

    case PI_CIV:

    case PO_CIV:

    case MC_CIV:

        r = &s->bm_regs[GET_BM (index)];

        val = r->civ | (r->lvi << 8) | (r->sr << 16);

        dolog ("CIV LVI SR[%d] -> %#x, %#x, %#x\n", GET_BM (index),

               r->civ, r->lvi, r->sr);

        break;

    case PI_PICB:

    case PO_PICB:

    case MC_PICB:

        r = &s->bm_regs[GET_BM (index)];

        val = r->picb | (r->piv << 16) | (r->cr << 24);

        dolog ("PICB PIV CR[%d] -> %#x %#x %#x %#x\n", GET_BM (index),

               val, r->picb, r->piv, r->cr);

        break;

    case GLOB_CNT:

        val = s->glob_cnt;

        dolog ("glob_cnt -> %#x\n", val);

        break;

    case GLOB_STA:

        val = s->glob_sta | GS_S0CR;

        dolog ("glob_sta -> %#x\n", val);

        break;

    default:

        dolog ("U nabm readl %#x -> %#x\n", addr, val);

        break;

    }

    return val;

}

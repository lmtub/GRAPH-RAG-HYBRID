static target_ulong compute_tlbie_rb(target_ulong v, target_ulong r,

                                     target_ulong pte_index)

{

    target_ulong rb, va_low;



    rb = (v & ~0x7fULL) << 16; /* AVA field */

    va_low = pte_index >> 3;

    if (v & HPTE64_V_SECONDARY) {

        va_low = ~va_low;

    }

    /* xor vsid from AVA */

    if (!(v & HPTE64_V_1TB_SEG)) {

        va_low ^= v >> 12;

    } else {

        va_low ^= v >> 24;

    }

    va_low &= 0x7ff;

    if (v & HPTE64_V_LARGE) {

        rb |= 1;                         /* L field */

#if 0 /* Disable that P7 specific bit for now */

        if (r & 0xff000) {

            /* non-16MB large page, must be 64k */

            /* (masks depend on page size) */

            rb |= 0x1000;                /* page encoding in LP field */

            rb |= (va_low & 0x7f) << 16; /* 7b of VA in AVA/LP field */

            rb |= (va_low & 0xfe);       /* AVAL field */

        }

#endif

    } else {

        /* 4kB page */

        rb |= (va_low & 0x7ff) << 12;   /* remaining 11b of AVA */

    }

    rb |= (v >> 54) & 0x300;            /* B field */

    return rb;

}

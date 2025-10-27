static void rtas_nvram_store(sPAPREnvironment *spapr,

                             uint32_t token, uint32_t nargs,

                             target_ulong args,

                             uint32_t nret, target_ulong rets)

{

    sPAPRNVRAM *nvram = spapr->nvram;

    hwaddr offset, buffer, len;

    int alen;

    void *membuf;



    if ((nargs != 3) || (nret != 2)) {

        rtas_st(rets, 0, -3);

        return;

    }



    if (!nvram) {

        rtas_st(rets, 0, -1);

        return;

    }



    offset = rtas_ld(args, 0);

    buffer = rtas_ld(args, 1);

    len = rtas_ld(args, 2);



    if (((offset + len) < offset)

        || ((offset + len) > nvram->size)) {

        rtas_st(rets, 0, -3);

        return;

    }



    membuf = cpu_physical_memory_map(buffer, &len, 0);

    if (nvram->drive) {

        alen = bdrv_pwrite(nvram->drive, offset, membuf, len);

    } else {

        assert(nvram->buf);



        memcpy(nvram->buf + offset, membuf, len);

        alen = len;

    }

    cpu_physical_memory_unmap(membuf, len, 0, len);



    rtas_st(rets, 0, (alen < len) ? -1 : 0);

    rtas_st(rets, 1, (alen < 0) ? 0 : alen);

}

int sclp_service_call(uint64_t sccb, uint32_t code)

{

    int r = 0;

    SCCB work_sccb;



    hwaddr sccb_len = sizeof(SCCB);



    /* first some basic checks on program checks */

    if (cpu_physical_memory_is_io(sccb)) {

        r = -PGM_ADDRESSING;

        goto out;

    }

    if (sccb & ~0x7ffffff8ul) {

        r = -PGM_SPECIFICATION;

        goto out;

    }



    /*

     * we want to work on a private copy of the sccb, to prevent guests

     * from playing dirty tricks by modifying the memory content after

     * the host has checked the values

     */

    cpu_physical_memory_read(sccb, &work_sccb, sccb_len);



    /* Valid sccb sizes */

    if (be16_to_cpu(work_sccb.h.length) < sizeof(SCCBHeader) ||

        be16_to_cpu(work_sccb.h.length) > SCCB_SIZE) {

        r = -PGM_SPECIFICATION;

        goto out;

    }



    sclp_execute((SCCB *)&work_sccb, code);



    cpu_physical_memory_write(sccb, &work_sccb,

                              be16_to_cpu(work_sccb.h.length));



    sclp_service_interrupt(sccb);



out:

    return r;

}

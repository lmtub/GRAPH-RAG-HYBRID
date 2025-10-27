static int ehci_state_fetchitd(EHCIState *ehci, int async)

{

    uint32_t entry;

    EHCIitd itd;



    assert(!async);

    entry = ehci_get_fetch_addr(ehci, async);



    get_dwords(NLPTR_GET(entry),(uint32_t *) &itd,

               sizeof(EHCIitd) >> 2);

    ehci_trace_itd(ehci, entry, &itd);



    if (ehci_process_itd(ehci, &itd) != 0) {

        return -1;

    }



    put_dwords(NLPTR_GET(entry), (uint32_t *) &itd,

                sizeof(EHCIitd) >> 2);

    ehci_set_fetch_addr(ehci, async, itd.next);

    ehci_set_state(ehci, async, EST_FETCHENTRY);



    return 1;

}

static bool xhci_er_full(void *opaque, int version_id)

{

    struct XHCIInterrupter *intr = opaque;

    return intr->er_full;

}

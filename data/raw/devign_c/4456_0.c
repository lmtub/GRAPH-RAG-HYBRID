int chsc_sei_nt2_have_event(void)

{

    S390pciState *s = S390_PCI_HOST_BRIDGE(

        object_resolve_path(TYPE_S390_PCI_HOST_BRIDGE, NULL));



    if (!s) {

        return 0;

    }



    return !QTAILQ_EMPTY(&s->pending_sei);

}

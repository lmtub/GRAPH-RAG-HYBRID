static gint ppc_cpu_compare_class_pvr_mask(gconstpointer a, gconstpointer b)

{

    ObjectClass *oc = (ObjectClass *)a;

    uint32_t pvr = *(uint32_t *)b;

    PowerPCCPUClass *pcc = (PowerPCCPUClass *)a;

    gint ret;



    /* -cpu host does a PVR lookup during construction */

    if (unlikely(strcmp(object_class_get_name(oc),

                        TYPE_HOST_POWERPC_CPU) == 0)) {

        return -1;

    }



    if (!ppc_cpu_is_valid(pcc)) {

        return -1;

    }



    ret = (((pcc->pvr & pcc->pvr_mask) == (pvr & pcc->pvr_mask)) ? 0 : -1);



    return ret;

}

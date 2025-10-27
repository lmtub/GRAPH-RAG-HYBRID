static void rtas_ibm_configure_connector(PowerPCCPU *cpu,

                                         sPAPRMachineState *spapr,

                                         uint32_t token, uint32_t nargs,

                                         target_ulong args, uint32_t nret,

                                         target_ulong rets)

{

    uint64_t wa_addr;

    uint64_t wa_offset;

    uint32_t drc_index;

    sPAPRDRConnector *drc;

    sPAPRDRConnectorClass *drck;

    sPAPRConfigureConnectorState *ccs;

    sPAPRDRCCResponse resp = SPAPR_DR_CC_RESPONSE_CONTINUE;

    int rc;

    const void *fdt;



    if (nargs != 2 || nret != 1) {

        rtas_st(rets, 0, RTAS_OUT_PARAM_ERROR);

        return;




    wa_addr = ((uint64_t)rtas_ld(args, 1) << 32) | rtas_ld(args, 0);



    drc_index = rtas_ld(wa_addr, 0);

    drc = spapr_dr_connector_by_index(drc_index);

    if (!drc) {

        DPRINTF("rtas_ibm_configure_connector: invalid DRC index: %xh\n",


        rc = RTAS_OUT_PARAM_ERROR;





    drck = SPAPR_DR_CONNECTOR_GET_CLASS(drc);

    fdt = drck->get_fdt(drc, NULL);









    ccs = spapr_ccs_find(spapr, drc_index);

    if (!ccs) {

        ccs = g_new0(sPAPRConfigureConnectorState, 1);

        (void)drck->get_fdt(drc, &ccs->fdt_offset);

        ccs->drc_index = drc_index;

        spapr_ccs_add(spapr, ccs);




    do {

        uint32_t tag;

        const char *name;

        const struct fdt_property *prop;

        int fdt_offset_next, prop_len;



        tag = fdt_next_tag(fdt, ccs->fdt_offset, &fdt_offset_next);



        switch (tag) {

        case FDT_BEGIN_NODE:

            ccs->fdt_depth++;

            name = fdt_get_name(fdt, ccs->fdt_offset, NULL);



            /* provide the name of the next OF node */

            wa_offset = CC_VAL_DATA_OFFSET;

            rtas_st(wa_addr, CC_IDX_NODE_NAME_OFFSET, wa_offset);

            rtas_st_buffer_direct(wa_addr + wa_offset, CC_WA_LEN - wa_offset,

                                  (uint8_t *)name, strlen(name) + 1);

            resp = SPAPR_DR_CC_RESPONSE_NEXT_CHILD;

            break;

        case FDT_END_NODE:

            ccs->fdt_depth--;

            if (ccs->fdt_depth == 0) {

                /* done sending the device tree, don't need to track

                 * the state anymore

                 */

                drck->set_configured(drc);

                spapr_ccs_remove(spapr, ccs);

                ccs = NULL;

                resp = SPAPR_DR_CC_RESPONSE_SUCCESS;

            } else {

                resp = SPAPR_DR_CC_RESPONSE_PREV_PARENT;


            break;

        case FDT_PROP:

            prop = fdt_get_property_by_offset(fdt, ccs->fdt_offset,

                                              &prop_len);

            name = fdt_string(fdt, fdt32_to_cpu(prop->nameoff));



            /* provide the name of the next OF property */

            wa_offset = CC_VAL_DATA_OFFSET;

            rtas_st(wa_addr, CC_IDX_PROP_NAME_OFFSET, wa_offset);

            rtas_st_buffer_direct(wa_addr + wa_offset, CC_WA_LEN - wa_offset,

                                  (uint8_t *)name, strlen(name) + 1);



            /* provide the length and value of the OF property. data gets

             * placed immediately after NULL terminator of the OF property's

             * name string

             */

            wa_offset += strlen(name) + 1,

            rtas_st(wa_addr, CC_IDX_PROP_LEN, prop_len);

            rtas_st(wa_addr, CC_IDX_PROP_DATA_OFFSET, wa_offset);

            rtas_st_buffer_direct(wa_addr + wa_offset, CC_WA_LEN - wa_offset,

                                  (uint8_t *)((struct fdt_property *)prop)->data,

                                  prop_len);

            resp = SPAPR_DR_CC_RESPONSE_NEXT_PROPERTY;

            break;

        case FDT_END:

            resp = SPAPR_DR_CC_RESPONSE_ERROR;

        default:

            /* keep seeking for an actionable tag */

            break;


        if (ccs) {

            ccs->fdt_offset = fdt_offset_next;


    } while (resp == SPAPR_DR_CC_RESPONSE_CONTINUE);



    rc = resp;

out:

    rtas_st(rets, 0, rc);

static void spapr_hotplug_req_event(uint8_t hp_id, uint8_t hp_action,

                                    sPAPRDRConnectorType drc_type,

                                    union drc_identifier *drc_id)

{

    sPAPRMachineState *spapr = SPAPR_MACHINE(qdev_get_machine());

    struct hp_log_full *new_hp;

    struct rtas_error_log *hdr;

    struct rtas_event_log_v6 *v6hdr;

    struct rtas_event_log_v6_maina *maina;

    struct rtas_event_log_v6_mainb *mainb;

    struct rtas_event_log_v6_hp *hp;



    new_hp = g_malloc0(sizeof(struct hp_log_full));

    hdr = &new_hp->hdr;

    v6hdr = &new_hp->v6hdr;

    maina = &new_hp->maina;

    mainb = &new_hp->mainb;

    hp = &new_hp->hp;



    hdr->summary = cpu_to_be32(RTAS_LOG_VERSION_6

                               | RTAS_LOG_SEVERITY_EVENT

                               | RTAS_LOG_DISPOSITION_NOT_RECOVERED

                               | RTAS_LOG_OPTIONAL_PART_PRESENT

                               | RTAS_LOG_INITIATOR_HOTPLUG

                               | RTAS_LOG_TYPE_HOTPLUG);

    hdr->extended_length = cpu_to_be32(sizeof(*new_hp)

                                       - sizeof(new_hp->hdr));



    spapr_init_v6hdr(v6hdr);

    spapr_init_maina(maina, 3 /* Main-A, Main-B, HP */);



    mainb->hdr.section_id = cpu_to_be16(RTAS_LOG_V6_SECTION_ID_MAINB);

    mainb->hdr.section_length = cpu_to_be16(sizeof(*mainb));

    mainb->subsystem_id = 0x80; /* External environment */

    mainb->event_severity = 0x00; /* Informational / non-error */

    mainb->event_subtype = 0x00; /* Normal shutdown */



    hp->hdr.section_id = cpu_to_be16(RTAS_LOG_V6_SECTION_ID_HOTPLUG);

    hp->hdr.section_length = cpu_to_be16(sizeof(*hp));

    hp->hdr.section_version = 1; /* includes extended modifier */

    hp->hotplug_action = hp_action;

    hp->hotplug_identifier = hp_id;



    switch (drc_type) {

    case SPAPR_DR_CONNECTOR_TYPE_PCI:

        hp->hotplug_type = RTAS_LOG_V6_HP_TYPE_PCI;

        if (hp->hotplug_action == RTAS_LOG_V6_HP_ACTION_ADD) {

            spapr_hotplug_set_signalled(drc_id->index);

        }

        break;

    case SPAPR_DR_CONNECTOR_TYPE_LMB:

        hp->hotplug_type = RTAS_LOG_V6_HP_TYPE_MEMORY;

        break;

    case SPAPR_DR_CONNECTOR_TYPE_CPU:

        hp->hotplug_type = RTAS_LOG_V6_HP_TYPE_CPU;

        break;

    default:

        /* we shouldn't be signaling hotplug events for resources

         * that don't support them

         */

        g_assert(false);

        return;

    }



    if (hp_id == RTAS_LOG_V6_HP_ID_DRC_COUNT) {

        hp->drc_id.count = cpu_to_be32(drc_id->count);

    } else if (hp_id == RTAS_LOG_V6_HP_ID_DRC_INDEX) {

        hp->drc_id.index = cpu_to_be32(drc_id->index);

    } else if (hp_id == RTAS_LOG_V6_HP_ID_DRC_COUNT_INDEXED) {

        /* we should not be using count_indexed value unless the guest

         * supports dedicated hotplug event source

         */

        g_assert(spapr_ovec_test(spapr->ov5_cas, OV5_HP_EVT));

        hp->drc_id.count_indexed.count =

            cpu_to_be32(drc_id->count_indexed.count);

        hp->drc_id.count_indexed.index =

            cpu_to_be32(drc_id->count_indexed.index);

    }



    rtas_event_log_queue(RTAS_LOG_TYPE_HOTPLUG, new_hp, true);



    qemu_irq_pulse(xics_get_qirq(XICS_FABRIC(spapr),

                                 rtas_event_log_to_irq(spapr,

                                                       RTAS_LOG_TYPE_HOTPLUG)));

}

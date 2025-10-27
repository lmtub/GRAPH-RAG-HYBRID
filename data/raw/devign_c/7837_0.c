void spapr_core_unplug_request(HotplugHandler *hotplug_dev, DeviceState *dev,

                               Error **errp)

{

    int index;

    sPAPRDRConnector *drc;

    sPAPRDRConnectorClass *drck;

    Error *local_err = NULL;

    CPUCore *cc = CPU_CORE(dev);

    int smt = kvmppc_smt_threads();



    if (!spapr_find_cpu_slot(MACHINE(hotplug_dev), cc->core_id, &index)) {

        error_setg(errp, "Unable to find CPU core with core-id: %d",

                   cc->core_id);

        return;

    }

    if (index == 0) {

        error_setg(errp, "Boot CPU core may not be unplugged");

        return;

    }



    drc = spapr_dr_connector_by_id(SPAPR_DR_CONNECTOR_TYPE_CPU, index * smt);

    g_assert(drc);



    drck = SPAPR_DR_CONNECTOR_GET_CLASS(drc);

    drck->detach(drc, dev, spapr_core_release, NULL, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }



    spapr_hotplug_req_remove_by_index(drc);

}

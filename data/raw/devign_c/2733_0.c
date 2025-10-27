void numa_cpu_pre_plug(const CPUArchId *slot, DeviceState *dev, Error **errp)

{

    int mapped_node_id; /* set by -numa option */

    int node_id = object_property_get_int(OBJECT(dev), "node-id", &error_abort);



    /* by default CPUState::numa_node was 0 if it wasn't set explicitly

     * TODO: make it error when incomplete numa mapping support is removed

     */

    mapped_node_id = slot->props.node_id;

    if (!slot->props.has_node_id) {

        mapped_node_id = 0;

    }



    if (node_id == CPU_UNSET_NUMA_NODE_ID) {

        /* due to bug in libvirt, it doesn't pass node-id from props on

         * device_add as expected, so we have to fix it up here */

        object_property_set_int(OBJECT(dev), mapped_node_id, "node-id", errp);

    } else if (node_id != mapped_node_id) {

        error_setg(errp, "node-id=%d must match numa node specified "

                   "with -numa option", node_id);

    }

}

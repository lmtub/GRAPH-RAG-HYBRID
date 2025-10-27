static void pc_cpu_pre_plug(HotplugHandler *hotplug_dev,

                            DeviceState *dev, Error **errp)

{

    int idx;

    int node_id;

    CPUState *cs;

    CPUArchId *cpu_slot;

    X86CPUTopoInfo topo;

    X86CPU *cpu = X86_CPU(dev);

    PCMachineState *pcms = PC_MACHINE(hotplug_dev);



    /* if APIC ID is not set, set it based on socket/core/thread properties */

    if (cpu->apic_id == UNASSIGNED_APIC_ID) {

        int max_socket = (max_cpus - 1) / smp_threads / smp_cores;



        if (cpu->socket_id < 0) {

            error_setg(errp, "CPU socket-id is not set");

            return;

        } else if (cpu->socket_id > max_socket) {

            error_setg(errp, "Invalid CPU socket-id: %u must be in range 0:%u",

                       cpu->socket_id, max_socket);

            return;

        }

        if (cpu->core_id < 0) {

            error_setg(errp, "CPU core-id is not set");

            return;

        } else if (cpu->core_id > (smp_cores - 1)) {

            error_setg(errp, "Invalid CPU core-id: %u must be in range 0:%u",

                       cpu->core_id, smp_cores - 1);

            return;

        }

        if (cpu->thread_id < 0) {

            error_setg(errp, "CPU thread-id is not set");

            return;

        } else if (cpu->thread_id > (smp_threads - 1)) {

            error_setg(errp, "Invalid CPU thread-id: %u must be in range 0:%u",

                       cpu->thread_id, smp_threads - 1);

            return;

        }



        topo.pkg_id = cpu->socket_id;

        topo.core_id = cpu->core_id;

        topo.smt_id = cpu->thread_id;

        cpu->apic_id = apicid_from_topo_ids(smp_cores, smp_threads, &topo);

    }



    cpu_slot = pc_find_cpu_slot(MACHINE(pcms), cpu->apic_id, &idx);

    if (!cpu_slot) {

        MachineState *ms = MACHINE(pcms);



        x86_topo_ids_from_apicid(cpu->apic_id, smp_cores, smp_threads, &topo);

        error_setg(errp, "Invalid CPU [socket: %u, core: %u, thread: %u] with"

                  " APIC ID %" PRIu32 ", valid index range 0:%d",

                   topo.pkg_id, topo.core_id, topo.smt_id, cpu->apic_id,

                   ms->possible_cpus->len - 1);

        return;

    }



    if (cpu_slot->cpu) {

        error_setg(errp, "CPU[%d] with APIC ID %" PRIu32 " exists",

                   idx, cpu->apic_id);

        return;

    }



    /* if 'address' properties socket-id/core-id/thread-id are not set, set them

     * so that machine_query_hotpluggable_cpus would show correct values

     */

    /* TODO: move socket_id/core_id/thread_id checks into x86_cpu_realizefn()

     * once -smp refactoring is complete and there will be CPU private

     * CPUState::nr_cores and CPUState::nr_threads fields instead of globals */

    x86_topo_ids_from_apicid(cpu->apic_id, smp_cores, smp_threads, &topo);

    if (cpu->socket_id != -1 && cpu->socket_id != topo.pkg_id) {

        error_setg(errp, "property socket-id: %u doesn't match set apic-id:"

            " 0x%x (socket-id: %u)", cpu->socket_id, cpu->apic_id, topo.pkg_id);

        return;

    }

    cpu->socket_id = topo.pkg_id;



    if (cpu->core_id != -1 && cpu->core_id != topo.core_id) {

        error_setg(errp, "property core-id: %u doesn't match set apic-id:"

            " 0x%x (core-id: %u)", cpu->core_id, cpu->apic_id, topo.core_id);

        return;

    }

    cpu->core_id = topo.core_id;



    if (cpu->thread_id != -1 && cpu->thread_id != topo.smt_id) {

        error_setg(errp, "property thread-id: %u doesn't match set apic-id:"

            " 0x%x (thread-id: %u)", cpu->thread_id, cpu->apic_id, topo.smt_id);

        return;

    }

    cpu->thread_id = topo.smt_id;



    cs = CPU(cpu);

    cs->cpu_index = idx;



    node_id = cpu_slot->props.node_id;

    if (!cpu_slot->props.has_node_id) {

        /* by default CPUState::numa_node was 0 if it's not set via CLI

         * keep it this way for now but in future we probably should

         * refuse to start up with incomplete numa mapping */

        node_id = 0;

    }

    if (cs->numa_node == CPU_UNSET_NUMA_NODE_ID) {

        cs->numa_node = node_id;

    } else if (cs->numa_node != node_id) {

            error_setg(errp, "node-id %d must match numa node specified"

                "with -numa option for cpu-index %d",

                cs->numa_node, cs->cpu_index);

            return;

    }

}

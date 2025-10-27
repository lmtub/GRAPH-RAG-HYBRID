static void read_SCP_info(SCLPDevice *sclp, SCCB *sccb)

{

    ReadInfo *read_info = (ReadInfo *) sccb;

    MachineState *machine = MACHINE(qdev_get_machine());

    sclpMemoryHotplugDev *mhd = get_sclp_memory_hotplug_dev();

    CPUState *cpu;

    int cpu_count = 0;

    int rnsize, rnmax;

    int slots = MIN(machine->ram_slots, s390_get_memslot_count(kvm_state));

    IplParameterBlock *ipib = s390_ipl_get_iplb();



    CPU_FOREACH(cpu) {

        cpu_count++;

    }



    /* CPU information */

    read_info->entries_cpu = cpu_to_be16(cpu_count);

    read_info->offset_cpu = cpu_to_be16(offsetof(ReadInfo, entries));

    read_info->highest_cpu = cpu_to_be16(max_cpus);



    read_info->ibc_val = cpu_to_be32(s390_get_ibc_val());



    /* Configuration Characteristic (Extension) */

    s390_get_feat_block(S390_FEAT_TYPE_SCLP_CONF_CHAR,

                         read_info->conf_char);

    s390_get_feat_block(S390_FEAT_TYPE_SCLP_CONF_CHAR_EXT,

                         read_info->conf_char_ext);



    prepare_cpu_entries(sclp, read_info->entries, cpu_count);



    read_info->facilities = cpu_to_be64(SCLP_HAS_CPU_INFO |

                                        SCLP_HAS_PCI_RECONFIG);



    /* Memory Hotplug is only supported for the ccw machine type */

    if (mhd) {

        mhd->standby_subregion_size = MEM_SECTION_SIZE;

        /* Deduct the memory slot already used for core */

        if (slots > 0) {

            while ((mhd->standby_subregion_size * (slots - 1)

                    < mhd->standby_mem_size)) {

                mhd->standby_subregion_size = mhd->standby_subregion_size << 1;

            }

        }

        /*

         * Initialize mapping of guest standby memory sections indicating which

         * are and are not online. Assume all standby memory begins offline.

         */

        if (mhd->standby_state_map == 0) {

            if (mhd->standby_mem_size % mhd->standby_subregion_size) {

                mhd->standby_state_map = g_malloc0((mhd->standby_mem_size /

                                             mhd->standby_subregion_size + 1) *

                                             (mhd->standby_subregion_size /

                                             MEM_SECTION_SIZE));

            } else {

                mhd->standby_state_map = g_malloc0(mhd->standby_mem_size /

                                                   MEM_SECTION_SIZE);

            }

        }

        mhd->padded_ram_size = ram_size + mhd->pad_size;

        mhd->rzm = 1 << mhd->increment_size;



        read_info->facilities |= cpu_to_be64(SCLP_FC_ASSIGN_ATTACH_READ_STOR);

    }

    read_info->mha_pow = s390_get_mha_pow();

    read_info->hmfai = cpu_to_be32(s390_get_hmfai());



    rnsize = 1 << (sclp->increment_size - 20);

    if (rnsize <= 128) {

        read_info->rnsize = rnsize;

    } else {

        read_info->rnsize = 0;

        read_info->rnsize2 = cpu_to_be32(rnsize);

    }



    rnmax = machine->maxram_size >> sclp->increment_size;

    if (rnmax < 0x10000) {

        read_info->rnmax = cpu_to_be16(rnmax);

    } else {

        read_info->rnmax = cpu_to_be16(0);

        read_info->rnmax2 = cpu_to_be64(rnmax);

    }



    if (ipib && ipib->flags & DIAG308_FLAGS_LP_VALID) {

        memcpy(&read_info->loadparm, &ipib->loadparm,

               sizeof(read_info->loadparm));

    } else {

        s390_ipl_set_loadparm(read_info->loadparm);

    }



    sccb->h.response_code = cpu_to_be16(SCLP_RC_NORMAL_READ_COMPLETION);

}

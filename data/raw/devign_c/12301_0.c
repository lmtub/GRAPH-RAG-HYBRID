static inline bool valid_ptex(PowerPCCPU *cpu, target_ulong ptex)

{

    /*

     * hash value/pteg group index is normalized by htab_mask

     */

    if (((ptex & ~7ULL) / HPTES_PER_GROUP) & ~cpu->env.htab_mask) {

        return false;

    }

    return true;

}

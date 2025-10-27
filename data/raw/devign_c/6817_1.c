void helper_movcal(CPUSH4State *env, uint32_t address, uint32_t value)

{

    if (cpu_sh4_is_cached (env, address))

    {

	memory_content *r = malloc (sizeof(memory_content));

	r->address = address;

	r->value = value;

	r->next = NULL;



	*(env->movcal_backup_tail) = r;

	env->movcal_backup_tail = &(r->next);

    }

}

void helper_discard_movcal_backup(CPUSH4State *env)

{

    memory_content *current = env->movcal_backup;



    while(current)

    {

	memory_content *next = current->next;

	free (current);

	env->movcal_backup = current = next;

	if (current == NULL)

	    env->movcal_backup_tail = &(env->movcal_backup);

    } 

}

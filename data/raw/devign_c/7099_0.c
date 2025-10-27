unsigned long setup_arg_pages(void * mh, char ** argv, char ** env)

{

    unsigned long stack_base, error, size;

    int i;

    int * stack;

    int argc, envc;



    /* Create enough stack to hold everything.  If we don't use

     * it for args, we'll use it for something else...

     */

    size = stack_size;



    error = target_mmap(0,

                        size + qemu_host_page_size,

                        PROT_READ | PROT_WRITE,

                        MAP_PRIVATE | MAP_ANONYMOUS,

                        -1, 0);

    if (error == -1)

        qerror("stk mmap");



    /* we reserve one extra page at the top of the stack as guard */

    target_mprotect(error + size, qemu_host_page_size, PROT_NONE);



    stack_base = error + size;

    stack = (void*)stack_base;

/*

 *    | STRING AREA |

 *    +-------------+

 *    |      0      |

*    +-------------+

 *    |  apple[n]   |

 *    +-------------+

 *           :

 *    +-------------+

 *    |  apple[0]   |

 *    +-------------+

 *    |      0      |

 *    +-------------+

 *    |    env[n]   |

 *    +-------------+

 *           :

 *           :

 *    +-------------+

 *    |    env[0]   |

 *    +-------------+

 *    |      0      |

 *    +-------------+

 *    | arg[argc-1] |

 *    +-------------+

 *           :

 *           :

 *    +-------------+

 *    |    arg[0]   |

 *    +-------------+

 *    |     argc    |

 *    +-------------+

 * sp->    |      mh     | address of where the a.out's file offset 0 is in memory

 *    +-------------+

*/

    /* Construct the stack Stack grows down */

    stack--;



    /* XXX: string should go up there */



    *stack = 0;

    stack--;



    /* Push the absolute path of our executable */

    DPRINTF("pushing apple %s (0x%x)\n", (char*)argv[0], (int)argv[0]);

    stl(stack, (int) argv[0]);



    stack--;



    stl(stack, 0);

    stack--;



    /* Get envc */

    for(envc = 0; env[envc]; envc++);



    for(i = envc-1; i >= 0; i--)

    {

        DPRINTF("pushing env %s (0x%x)\n", (char*)env[i], (int)env[i]);

        stl(stack, (int)env[i]);

        stack--;



        /* XXX: remove that when string will be on top of the stack */

        page_set_flags((int)env[i], (int)(env[i]+strlen(env[i])), PROT_READ | PAGE_VALID);

    }



    /* Add on the stack the interp_prefix choosen if so */

    if(interp_prefix[0])

    {

        char *dyld_root;

        asprintf(&dyld_root, "DYLD_ROOT_PATH=%s", interp_prefix);

        page_set_flags((int)dyld_root, (int)(dyld_root+strlen(interp_prefix)+1), PROT_READ | PAGE_VALID);



        stl(stack, (int)dyld_root);

        stack--;

    }



#ifdef DONT_USE_DYLD_SHARED_MAP

    {

        char *shared_map_mode;

        asprintf(&shared_map_mode, "DYLD_SHARED_REGION=avoid");

        page_set_flags((int)shared_map_mode, (int)(shared_map_mode+strlen(shared_map_mode)+1), PROT_READ | PAGE_VALID);



        stl(stack, (int)shared_map_mode);

        stack--;

    }

#endif



#ifdef ACTIVATE_DYLD_TRACE

    char * extra_env_static[] = {"DYLD_DEBUG_TRACE=yes",

    "DYLD_PREBIND_DEBUG=3", "DYLD_UNKNOW_TRACE=yes",

    "DYLD_PRINT_INITIALIZERS=yes",

    "DYLD_PRINT_SEGMENTS=yes", "DYLD_PRINT_REBASINGS=yes", "DYLD_PRINT_BINDINGS=yes", "DYLD_PRINT_INITIALIZERS=yes", "DYLD_PRINT_WARNINGS=yes" };



    char ** extra_env = malloc(sizeof(extra_env_static));

    bcopy(extra_env_static, extra_env, sizeof(extra_env_static));

    page_set_flags((int)extra_env, (int)((void*)extra_env+sizeof(extra_env_static)), PROT_READ | PAGE_VALID);



    for(i = 0; i<9; i++)

    {

        DPRINTF("pushing (extra) env %s (0x%x)\n", (char*)extra_env[i], (int)extra_env[i]);

        stl(stack, (int) extra_env[i]);

        stack--;

    }

#endif



    stl(stack, 0);

    stack--;



    /* Get argc */

    for(argc = 0; argv[argc]; argc++);



    for(i = argc-1; i >= 0; i--)

    {

        DPRINTF("pushing arg %s (0x%x)\n", (char*)argv[i], (int)argv[i]);

        stl(stack, (int) argv[i]);

        stack--;



        /* XXX: remove that when string will be on top of the stack */

        page_set_flags((int)argv[i], (int)(argv[i]+strlen(argv[i])), PROT_READ | PAGE_VALID);

    }



    DPRINTF("pushing argc %d \n", argc);

    stl(stack, argc);

    stack--;



    DPRINTF("pushing mh 0x%x \n", (int)mh);

    stl(stack, (int) mh);



    /* Stack points on the mh */

    return (unsigned long)stack;

}

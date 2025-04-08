project('RLforHFT', 'cpp',
        version : '1.0.0',
        default_options : ['warning_level=3', 'cpp_std=c++20'])


eigen_dep = dependency('eigen3')
git = find_program('git')
cpp = meson.get_compiler('cpp')
python3 = import('python').find_installation()
fs = import('fs')
ros_comm_dep = dependency('ros_comm', required : false)
oepnmp_dep = dependency('openmp')

sdl_dep = dependency('sdl', fallback : ['sdl', 'sdl_dep'])



if not ros_comm_dep.found()
    # 디렉토리가 이미 존재하는지 확인
    if not fs.exists('ros_comm')
        clone_result = run_command(git, 'clone',
                                   'https://github.com/ros/ros_comm.git',
                                   check : false)

        if clone_result.returncode() != 0
            error('Failed to clone ros_comm repository')
        endif
    endif
endif
mlpack_dep = dependency('mlpack', required : false)



# If not found as a system dependency, handle it manually
if not mlpack_dep.found()
    # You might need to specify include directories manually
    mlpack_inc = include_directories('mlpack/include')

    # And link against the library
    mlpack_lib = compiler.find_library('mlpack', required : true)

    # Combine them into a dependency object
    mlpack_dep = declare_dependency(
        include_directories : mlpack_inc,
        dependencies : mlpack_lib
    )
endif


src = ['main.cpp']
LLSrc = ['LowLatency/tcp_server.cpp',
         'LowLatency/tcp_socket.cpp', 'LowLatency/mcast_socket.cpp', 'LowLatency/me_order_book.cpp',
         'LowLatency/exchange_main.cpp', 'LowLatency/matching_engine.cpp', 'LowLatency/me_order.cpp'
         , 'LowLatency/order_server.cpp', 'LowLatency/snapshot_synthesizer.cpp', 'LowLatency/market_data_publisher.cpp',
         'LowLatency/position_keeper.cpp', 'LowLatency/market_order_book.cpp', 'LowLatency/market_order.cpp'

]

MLSrc = ['ML/main.cpp',
         'ML/SimpleRL.cpp',
         'ML/abstract_network_node.cpp',
         'ML/abstract_merge_node.cpp',
         'ML/abstract_recurrent_network_node.cpp',
         'ML/dense.cpp',
         'ML/state_aggregation.cpp',
         'ML/network_serializer.cpp',
         'ML/network.cpp',
         'ML/merge_sum.cpp',
         'ML/merge_product.cpp',
         'ML/lstm.cpp',
         'ML/gru.cpp',
         'ML/cw_rnn.cpp',
         'ML/tile_coding.cpp',
         'ML/sarsa.cpp', 'ML/epsilon_greedy.cpp', 'ML/flappy_simulator.cpp',
         'ML/ppo2.cpp']

MLSrc2 = ['ML2/data_process.cpp']

ModelingDerivativesSrc = ['Modeling Derivatives/main.cpp',
                          'Modeling Derivatives/matrix.cpp',
]
#==================================================================================================
incdirLL = include_directories('LowLatency/include')
incdirML = include_directories('ML/include')
incdirModelingDerivatives = include_directories('Modeling Derivatives/include')
incdirML2 = include_directories('ML2/include')
#==================================================================================================
libraryML = library('ML', MLSrc, include_directories : incdirML)
libraryModelingDerivatives = library('Modeling Derivatives',
                                     ModelingDerivativesSrc,
                                     include_directories : incdirModelingDerivatives)

libraryLL = library('LowLatency', LLSrc, include_directories : incdirLL)

libraryML2 = library('ML2', LLSrc, include_directories : incdirML2)
#==================================================================================================
RLforHFT = executable('RLforHFT', src, install : true,
                      link_with : [
                          libraryML,
                          libraryModelingDerivatives, libraryLL, libraryML2
                      ],
                      include_directories : [incdirML, incdirModelingDerivatives, incdirLL, incdirML2],
                      dependencies : [
                          eigen_dep,
                          ros_comm_dep,
                          oepnmp_dep,
                          sdl_dep, mlpack_dep
                      ])

test('test', RLforHFT)
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

src = ['main.cpp']

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
         'ML/sarsa.cpp', 'ML/epsilon_greedy.cpp',
         'ML/ppo2.cpp']

ModelingDerivativesSrc = ['Modeling Derivatives/main.cpp',
                          'Modeling Derivatives/matrix.cpp']

incdirML = include_directories('ML/include')
incdirModelingDerivatives = include_directories('Modeling Derivatives/include')

libraryML = library('ML', MLSrc, include_directories : incdirML)
libraryModelingDerivatives = library('Modeling Derivatives',
                                     ModelingDerivativesSrc,
                                     include_directories : incdirModelingDerivatives)


RLforHFT = executable('RLforHFT', src, install : true,
                      link_with : [
                          libraryML,
                          libraryModelingDerivatives],
                      include_directories : [incdirML, incdirModelingDerivatives],
                      dependencies : [
                          eigen_dep,
                          ros_comm_dep,
                          oepnmp_dep])

test('test', RLforHFT)
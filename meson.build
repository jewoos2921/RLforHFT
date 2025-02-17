project('RLforHFT', 'cpp',
        version : '1.0.0',
        default_options : ['warning_level=3', 'cpp_std=c++20'])


eigen_dep = dependency('eigen3')
oepnmp_dep = dependency('openmp')
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
#                          ros_comm_dep,
                          oepnmp_dep])

test('test', RLforHFT)

################################################################################
#                                                                              #
#	UJJWAL KHANDELWAL                                                          #    
#	PSO (PARTICLE SWARM OPTIMIZATION)                                          #
#	PYTHON 3.7.10                                                              #
#                                                                              #
################################################################################

######################## FITNESS FUNCTION 1 ######################################

def fitness_1(X):

    '''

    X: POSITION (EITHER CURRENT, LOCAL BEST OR GLOBAL BEST) OF SIZE (n,)

    EXAMPLE PRESENT IN [7] FOR 2-DIMENSIONAL VECTORS (X = (x,y))

    #################################################################################

    HIMMELBLAU'S FUNCTION

    MINIMIZE f(x) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
    
    OPTIMUM SOLUTION IS x* = 3 AND y* = 2

    REPLACE 'f' BELOW WITH THIS TO TEST EXAMPLE-1

    f = (x**2 + y - 11)**2 + (x + y**2 - 7)**2

    '''

    x, y = X[0][0], X[1][0]
    f = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    return f

######################## FITNESS FUNCTION 2 ######################################

def fitness_2(X):

    '''

    X: POSITION (EITHER CURRENT, LOCAL BEST OR GLOBAL BEST) OF SIZE (n,)

    EXAMPLE PRESENT IN [7] FOR 2-DIMENSIONAL VECTORS (X = (x,y))

    #################################################################################

    BOOTH'S FUNCTION

    MINIMIZE f(x) = (x + 2y - 7)^2 + (2x + y - 5)^2

    OPTIMUM SOLUTION IS x* = 1 AND y* = 3

    REPLACE 'f' BELOW WITH THIS TO TEST EXAMPLE-2

    f = (x + 2*y - 7)**2 + (2*x + y - 5)**2

    '''

    x, y = X[0][0], X[1][0]
    f = (x + 2*y - 7)**2 + (2*x + y - 5)**2
    return f

######################## FITNESS FUNCTION 3 ######################################

def fitness_3(X):

    '''

    X: POSITION (EITHER CURRENT, LOCAL BEST OR GLOBAL BEST) OF SIZE (n,)

    EXAMPLE PRESENT IN [7] FOR 2-DIMENSIONAL VECTORS (X = (x,y))

    #################################################################################

    BEALE'S FUNCTION

    MINIMIZE f(x) = (1.5 - x - xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2

    OPTIMUM SOLUTION IS x* = 3 AND y* = 0.5

    REPLACE 'f' BELOW WITH THIS TO TEST EXAMPLE-3

    f = (1.5 - x + x*y)**2 + (2.25 - x + x*(y**2))**2 + (2.625 - x + x*(y**3))**2
    
    #################################################################################

    '''

    x, y = X[0][0], X[1][0]
    f = (1.5 - x + x*y)**2 + (2.25 - x + x*(y**2))**2 + (2.625 - x + x*(y**3))**2
    return f

def fitness_4(X):

    '''

    X: POSITION (EITHER CURRENT, LOCAL BEST OR GLOBAL BEST) OF SIZE (n,)

    EXAMPLE PRESENT IN [https://www.ece.mcmaster.ca/~xwu/part4.pdf, pg. 14] FOR 2-DIMENSIONAL VECTORS (X = (x,y))

    #################################################################################

    MAXIMIZE f(x) = 2xy + 2x - x^2 - 2y^2

    OPTIMUM SOLUTION IS x* = 2 AND y* = 1

    REPLACE 'f' BELOW WITH THIS TO TEST fitness_4

    f = 2*x*y + 2*x - x**2 - 2*(y**2)
    
    #################################################################################

    '''
    x, y = X[0][0], X[1][0]
    f = 2*x*y + 2*x - x**2 - 2*(y**2)
    return f

#########################################################################################################################################
#                                                                                                                                       #
# REFERENCES:                                                                                                                           #
#                                                                                                                                       #
# [1] ALMEIDA, BRUNO & COPPO LEITE, VICTOR. (2019). PARTICLE SWARM OPTIMIZATION: A POWERFUL TECHNIQUE FOR                               #
#     SOLVING ENGINEERING PROBLEMS. 10.5772/INTECHOPEN.89633.                                                                           #
#                                                                                                                                       #
# [2] HE, YAN & MA, WEI & ZHANG, JI. (2016). THE PARAMETERS SELECTION OF PSO ALGORITHM INFLUENCING ON PERFORMANCE OF FAULT DIAGNOSIS.   #
#     MATEC WEB OF CONFERENCES. 63. 02019. 10.1051/MATECCONF/20166302019.                                                               #
#                                                                                                                                       #
# [3] CLERC, M., AND J. KENNEDY. THE PARTICLE SWARM — EXPLOSION, STABILITY, AND CONVERGENCE IN A MULTIDIMENSIONAL COMPLEX SPACE.        #
#     IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION 6, NO. 1 (FEBRUARY 2002): 58–73.                                                    #
#                                                                                                                                       #
# [4] Y. H. SHI AND R. C. EBERHART, “A MODIFIED PARTICLE SWARM OPTIMIZER,” IN PROCEEDINGS OF THE IEEE INTERNATIONAL                     #
#     CONFERENCES ON EVOLUTIONARY COMPUTATION, PP. 69–73, ANCHORAGE, ALASKA, USA, MAY 1998.                                             #
#                                                                                                                                       #
# [5] G. SERMPINIS, K. THEOFILATOS, A. KARATHANASOPOULOS, E. F. GEORGOPOULOS, & C. DUNIS, FORECASTING FOREIGN EXCHANGE                  #
#     RATES WITH ADAPTIVE NEURAL NETWORKS USING RADIAL-BASIS FUNCTIONS AND PARTICLE SWARM OPTIMIZATION,                                 #
#     EUROPEAN JOURNAL OF OPERATIONAL RESEARCH.                                                                                         #
#                                                                                                                                       #
# [6] PARTICLE SWARM OPTIMIZATION (PSO) VISUALLY EXPLAINED                                                                              #
#     (https://towardsdatascience.com/particle-swarm-optimization-visually-explained-46289eeb2e14)                                      #
#                                                                                                                                       #
# [7] RAJIB KUMAR BHATTACHARJYA, INTRODUCTION TO PARTICLE SWARM OPTIMIZATION                                                            #
#     (http://www.iitg.ac.in/rkbc/CE602/CE602/Particle%20Swarm%20Algorithms.pdf)                                                        #
#                                                                                                                                       #
#########################################################################################################################################

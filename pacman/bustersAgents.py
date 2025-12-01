# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import inference

import os
import sys
import random
from distanceCalculator import Distancer
from game import Actions
import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import busters
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

class NullGraphics:
    "Placeholder for graphics"

    def initialize(self, state, isBlue=False):
        pass

    def update(self, state):
        pass

    def pause(self):
        pass

    def draw(self, state):
        pass

    def updateDistributions(self, dist):
        pass

    def finish(self):
        pass


class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """

    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent:
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__(
            self,
            index=0,
            inference= "ExactInference",
            ghostAgents=None,
            observeEnable=True,
            elapseTimeEnable=True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution()
                             for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + \
            [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        # for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        # self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP


class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(
            self,
            index=0,
            inference="KeyboardInference",
            ghostAgents=None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def registerInitialState(self, gameState):
        self.countActions = 0
        self.comida_inicial = gameState.getNumFood()
        BustersAgent.registerInitialState(self, gameState)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        self.countActions += 1
        return KeyboardAgent.getAction(self, gameState)
    
    def final(self, gameState):
        print("\n=== FIN DE LA PARTIDA ===")
        print("Score final:", gameState.getScore())

        # Distancias a los fantasmas
        ghost_distances = gameState.data.ghostDistances
        fantasmas_comidos = sum(1 for d in ghost_distances if d == 0)
        print("Fantasmas comidos (distancia = 0):", fantasmas_comidos)

        comidas_restantes = gameState.getNumFood()
        print("Comida restante:", comidas_restantes)

        if hasattr(self, "comida_inicial"):
            comidas_comidas = self.comida_inicial - comidas_restantes
            print("Comidas comidas:", comidas_comidas)

        print("Ticks transcurridos:", self.countActions)


'''Random PacMan Agent'''


class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    ''' Example of counting something'''

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if (height):
                    food = food + 1
        return food

    ''' Print the layout'''

    def printGrid(self, gameState):
        table = ""
        # print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + \
                    gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0)  # Legal position from the pacman
        move_random = random.randint(0, 3)
        if (move_random == 0) and Directions.WEST in legal:
            move = Directions.WEST
        if (move_random == 1) and Directions.EAST in legal:
            move = Directions.EAST
        if (move_random == 2) and Directions.NORTH in legal:
            move = Directions.NORTH
        if (move_random == 3) and Directions.SOUTH in legal:
            move = Directions.SOUTH
        return move


class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i + 1]]
        return Directions.EAST


class BasicAgentAA(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0

    ''' Example of counting something'''

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if (height):
                    food = food + 1
        return food

    ''' Print the layout'''

    def printGrid(self, gameState):
        table = ""
        # print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + \
                    gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print(
            "---------------- TICK ",
            self.countActions,
            " --------------------------")
        # Dimensiones del mapa
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Posicion del Pacman
        print("Pacman position: ", gameState.getPacmanPosition())
        # Acciones legales de pacman en la posicion actual
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Direccion de pacman
        print(
            "Pacman direction: ",
            gameState.data.agentStates[0].getDirection())
        # Numero de fantasmas
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        # Fantasmas que estan vivos (el indice 0 del array que se devuelve
        # corresponde a pacman y siempre es false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Posicion de los fantasmas
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Direciones de los fantasmas
        print(
            "Ghosts directions: ", [
                gameState.getGhostDirections().get(i) for i in range(
                    0, gameState.getNumAgents() - 1)])
        # Distancia de manhattan a los fantasmas
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Puntos de comida restantes
        print("Pac dots: ", gameState.getNumFood())
        # Distancia de manhattan a la comida mas cercada
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood()[0])
        # Posición de la comida más cercana
        print("Position of nearest pac dot: ", gameState.getDistanceNearestFood()[1])
        # Paredes del mapa
        print("Map:  \n", gameState.getWalls())
        # Puntuacion
        print("Score: ", gameState.getScore())

    def chooseAction(self, gameState):
        self.countActions += 1
        self.printInfo(gameState)

        # Obter as accións legais do Pacman
        accions_legais = gameState.getLegalPacmanActions()
        
        # Obter a posición do Pacman
        posicion_pacman = gameState.getPacmanPosition()
        
        # Obter a posición dos fantasmas
        posicion_fantasmas = gameState.getGhostPositions()
        
        # Obter as distancias aos fantasmas
        distancias_fantasmas = gameState.data.ghostDistances
        
        # Obter a distancia e posición da comida máis próxima
        distancia_comida_proxima, posicion_comida_proxima = gameState.getDistanceNearestFood()
        
        # Lista de obxectivos ordenados por distancia
        obxectivos_ordenados = []

        # Engadir fantasmas á lista
        for i, distancia_fantasma in enumerate(distancias_fantasmas):
            if distancia_fantasma is not None:
                obxectivos_ordenados.append((distancia_fantasma, posicion_fantasmas[i]))
        
        # Engadir comida á lista
        if distancia_comida_proxima is not None:
            obxectivos_ordenados.append((distancia_comida_proxima, posicion_comida_proxima))
        
        # Ordenar os obxectivos por distancia (do máis próximo ao máis lonxano)
        obxectivos_ordenados.sort()

        # Intentar moverse cara ao obxectivo máis próximo dispoñible evitando paredes
        movemento = Directions.STOP
        for _, obxectivo in obxectivos_ordenados[:3]:  # Consideramos ata o segundo máis próximo
            if obxectivo[1] > posicion_pacman[1] and Directions.NORTH in accions_legais:
                movemento = Directions.NORTH
            elif obxectivo[1] < posicion_pacman[1] and Directions.SOUTH in accions_legais:
                movemento = Directions.SOUTH
            elif obxectivo[0] > posicion_pacman[0] and Directions.EAST in accions_legais:
                movemento = Directions.EAST
            elif obxectivo[0] < posicion_pacman[0] and Directions.WEST in accions_legais:
                movemento = Directions.WEST
            
            # Se atopamos un movemento válido, saímos do bucle
            if movemento != Directions.STOP:
                break

        # Se non hai movemento válido, escoller outro movemento aleatorio entre os legais
        if movemento == Directions.STOP:
            movemento = random.choice(accions_legais)
        
        return movemento

import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from scipy.io import arff as scipy_arff
import matplotlib.pyplot as plt

class AutomaticAgent(BustersAgent):
    PRUEBA_NUM = 2

    def calcular_deltas(self, px, py, ghost_coords):
        diffs = [(gx - px, gy - py) for gx, gy in ghost_coords]
        dists = [np.linalg.norm([dx, dy]) for dx, dy in diffs]
        min_idx = np.argmin(dists)
        dx, dy = diffs[min_idx]
        return dx, dy, dists[min_idx]

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)

        self.countActions = 0

        # Guardar número de comidas iniciales
        self.comida_inicial = gameState.getNumFood()

        # Cargar centroides
        ruta_resultados = f"ficheros_ejemplos/Prueva_{self.PRUEBA_NUM}_Resultados.txt"
        self.centroids = self.load_centroids(ruta_resultados)

        # ENTRENAR NORMALIZADOR desde el archivo .arff original
        ruta_arff_inicial = f"ficheros_ejemplos/Prueva_{self.PRUEBA_NUM}_Inicial.arff"
        data, _ = scipy_arff.loadarff(ruta_arff_inicial)
        df = pd.DataFrame(data)
        df["action"] = df["action"].apply(lambda x: x.decode() if isinstance(x, bytes) else x)

        px = df['pacman_x'].astype(float)
        py = df['pacman_y'].astype(float)
        ghost_coords = [
            (df[f'ghost{g}_x'].astype(float), df[f'ghost{g}_y'].astype(float))
            for g in range(1, 5)
        ]

        dxs, dys, acciones = [], [], []
        for idx in range(len(df)):
            ghosts = [(gx[idx], gy[idx]) for gx, gy in ghost_coords]
            dx, dy, _ = self.calcular_deltas(px[idx], py[idx], ghosts)
            dxs.append(dx)
            dys.append(dy)
            acciones.append(df["action"][idx])

        result_df = pd.DataFrame({
            'dist_nearest_ghost_x': dxs,
            'dist_nearest_ghost_y': dys,
            'action': acciones
        })

        result_df = result_df[result_df["action"] != "Stop"].reset_index(drop=True)
        self.instances = result_df

        self.scaler = MinMaxScaler()
        self.scaler.fit(result_df[["dist_nearest_ghost_x", "dist_nearest_ghost_y"]])

    def load_centroids(self, path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        start = next(i for i, l in enumerate(lines) if "Final cluster centroids" in l)
        attr_line = next(i for i, l in enumerate(lines[start:], start=start) if "dist_nearest_ghost_x" in l)
        x_line = lines[attr_line].strip().split()
        y_line = lines[attr_line + 1].strip().split()

        centroids = []
        for i in range(2, 6):
            try:
                x = float(x_line[i])
                y = float(y_line[i])
                centroids.append([x, y])
            except:
                continue

        if not centroids:
            print(f"No se encontraron centroides válidos en {path}")
        else:
            print(f"Centroides cargados correctamente desde {path}: {len(centroids)} clusters")

        return np.array(centroids)

    def euclidean_distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def chooseAction(self, gameState):
        self.countActions += 1

        if len(self.centroids) == 0:
            raise ValueError("No hay centroides cargados.")

        pacman_x, pacman_y = gameState.getPacmanPosition()
        ghost_positions = gameState.getGhostPositions()
        legal_actions = gameState.getLegalPacmanActions()

        distances = []
        for gx, gy in ghost_positions[:4]:
            dx = gx - pacman_x
            dy = gy - pacman_y
            distances.append((dx, dy, np.linalg.norm([dx, dy])))

        dx, dy, _ = min(distances, key=lambda x: x[2])
        estado_df = pd.DataFrame([[dx, dy]], columns=["dist_nearest_ghost_x", "dist_nearest_ghost_y"])
        dx_norm, dy_norm = self.scaler.transform(estado_df)[0]
        estado_actual = [dx_norm, dy_norm]

        cluster_id = np.argmin([self.euclidean_distance(estado_actual, c) for c in self.centroids])

        def asignar_cluster(row):
            punto = [row["dist_nearest_ghost_x"], row["dist_nearest_ghost_y"]]
            return np.argmin([self.euclidean_distance(punto, c) for c in self.centroids])

        self.instances["cluster_id"] = self.instances.apply(asignar_cluster, axis=1)
        instancias_cluster = self.instances[self.instances["cluster_id"] == cluster_id].copy()

        instancias_cluster["distancia"] = instancias_cluster.apply(
            lambda row: self.euclidean_distance(
                [row["dist_nearest_ghost_x"], row["dist_nearest_ghost_y"]],
                estado_actual
            ), axis=1
        )

        acciones = instancias_cluster["action"]
        accion = acciones.sample(weights=acciones.map(acciones.value_counts())).iloc[0]

        if accion not in legal_actions or accion == "Stop":
            legales = [a for a in legal_actions if a != "Stop"]
            accion = random.choice(legales) if legales else "North"

        self.visualizar_estado(estado_actual, cluster_id, instancias_cluster)
        return accion

    def final(self, gameState):
        print("\n=== FIN DE LA PARTIDA ===")
        print("Score final:", gameState.getScore())

        comidas_restantes = gameState.getNumFood()
        comidas_comidas = self.comida_inicial - comidas_restantes
        print("Comidas comidas:", comidas_comidas)

        print("Ticks transcurridos:", self.countActions)

    def visualizar_estado(self, estado, cluster_id, instancias_cluster):
        try:
            '''plt.clf()
            colors = ['blue', 'green', 'orange', 'purple']
            cluster_acciones = {}
            for i in range(len(self.centroids)):
                insts = self.instances[self.instances["cluster_id"] == i]
                if not insts.empty:
                    accion = insts["action"].value_counts().idxmax()
                else:
                    accion = "?"
                cluster_acciones[i] = accion

            for i, centroide in enumerate(self.centroids):
                label = f"Cluster {i}: {cluster_acciones[i]}"
                plt.scatter(*centroide, color=colors[i], marker='X', s=100, label=label)

            plt.scatter(estado[0], estado[1], color='red', marker='o', s=80, label="Estado actual")
            plt.title(f"Cluster seleccionado: {cluster_id}")
            plt.xlabel("dist_nearest_ghost_x")
            plt.ylabel("dist_nearest_ghost_y")
            plt.legend()
            plt.pause(0.01)'''
        except Exception as e:
            print(f"[DEBUG] Visualización no disponible: {e}")
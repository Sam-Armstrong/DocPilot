import numpy as np

def nearest_neighbor_tsp(cities):
    num_cities = len(cities)
    if num_cities < 2:
        return cities

    # Initialize variables
    unvisited_cities = set(range(num_cities))
    tour = [0]  # Start with the first city as the initial city
    current_city = 0

    while unvisited_cities:
        nearest_city = min(unvisited_cities, key=lambda city: np.linalg.norm(cities[current_city] - cities[city]))
        tour.append(nearest_city)
        current_city = nearest_city
        unvisited_cities.remove(nearest_city)

    # Return to the starting city to complete the tour
    tour.append(tour[0])

    return [cities[i] for i in tour]

# Example usage:
cities = np.array([[0, 0], [2, 4], [3, 1], [5, 3], [6, 5]])
optimal_tour = nearest_neighbor_tsp(cities)
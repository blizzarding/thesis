using Random
using DataFrames
using JuMP
using Ipopt  # Optimization solver
using CSV
using Distributions

# Set the seed for reproducibility
Random.seed!(0)

# Experience distribution data
experience_distribution = Dict(
    "<1 Year" => 12.94, "1 Year" => 11.03, "2 Years" => 8.64,
    "3 Years" => 6.61, "4 Years" => 5.24, "5 Years" => 3.69,
    "6-10 Years" => 7.26, "11-15 Years" => 4.23, "16-20 Years" => 15.31,
    "21-25 Years" => 15.39, "26-30 Years" => 6.93, "31-35 Years" => 1.94,
    "36-40 Years" => 0.66, ">40 Years" => 0.23
)

total_teachers = 2000
teacher_counts = Dict(key => round(Int, total_teachers * percent / 100) for (key, percent) in experience_distribution)
teacher_counts["<1 Year"] += total_teachers - sum(values(teacher_counts))

# Experience years ranges for categories
experience_years_ranges = Dict(
    "<1 Year" => (0, 1), "1 Year" => (1, 1), "2 Years" => (2, 2),
    "3 Years" => (3, 3), "4 Years" => (4, 4), "5 Years" => (5, 5),
    "6-10 Years" => (6, 10), "11-15 Years" => (11, 15),
    "16-20 Years" => (16, 20), "21-25 Years" => (21, 25),
    "26-30 Years" => (26, 30), "31-35 Years" => (31, 35),
    "36-40 Years" => (36, 40), ">40 Years" => (41, 45)
)

# Degree levels and distribution
degree_levels = ["HS", "BA", "MA", "MA+30", "Doctorate"]
degree_distribution = [0.010,0.2800,0.595,0.1036,0.0114]
degree_level_mapping = Dict("HS" => 0, "BA" => 1, "MA" => 2, "MA+30" => 3, "Doctorate" => 4)
degree_dist = Categorical(degree_distribution)

# Generate teacher data
teacher_data = DataFrame(ExperienceYears = Int[], DegreeLevel = String[], DegreeLevelNumeric = Int[])
for (category, num_teachers) in teacher_counts
    exp_range = experience_years_ranges[category]
    for _ in 1:num_teachers
        exp_years = rand(exp_range[1]:exp_range[2])
        degree_level = degree_levels[rand(degree_dist)]
        push!(teacher_data, (exp_years, degree_level, degree_level_mapping[degree_level]))
    end
end

# Optimization model setup
model = Model(Ipopt.Optimizer)
# set_silent(model)

@variable(model, 100000 >= b >= 30000)  # Base salary
@variable(model, 20000 >= c >= 3000)  # Scale factor
@variable(model, alpha >= 0, start = 0.5)  # Experience parameter (positive to ensure increasing salary with experience)
@variable(model, beta >= 0, start = 0.1)  # Degree level parameter (positive to ensure increasing salary with degree level)
@variable(model, 0 <= delta <= 1, start = 0.5)  # Effectiveness weight
@variable(model, 0 <= p_exp <= 3, start = 0.5)
@variable(model, 0 <= p_deg <= 3, start = 0.5)

# Constants for effectiveness calculation
total_budget = 242500000
rho = 0.5

# Objective function to maximize total effectiveness, ensuring it is concave and increasing
@NLobjective(model, Max, sum((delta * (b + c * (alpha * (max(teacher_data.ExperienceYears[i], 1e-6))^p_exp + beta * (max(teacher_data.DegreeLevelNumeric[i], 1e-6))^p_deg))^rho +
                               (1 - delta) * max(teacher_data.ExperienceYears[i], 1e-6)^rho)^(1/rho) for i in 1:nrow(teacher_data)))

# Use the full budget exactly
@NLconstraint(model, sum(b + c * (alpha * (max(teacher_data.ExperienceYears[i], 1e-6))^p_exp + beta * (max(teacher_data.DegreeLevelNumeric[i], 1e-6))^p_deg) for i in 1:nrow(teacher_data)) == value(total_budget))

# Solve the optimization problem
optimize!(model)

optimized_total_salary = sum(value(b) + value(c) * (value(alpha) * (max(1e-6,teacher_data.ExperienceYears[i]))^value(p_exp) + value(beta) * (max(1e-6,teacher_data.DegreeLevelNumeric[i]))^value(p_deg)) for i in 1:nrow(teacher_data))
optimized_total_effectiveness = sum(
    (value(delta) * (value(b) + value(c) * (value(alpha) * (max(1e-6,teacher_data.ExperienceYears[i]))^value(p_exp) + value(beta) * (max(1e-6,teacher_data.DegreeLevelNumeric[i]))^value(p_deg)))^rho +
                               (1 - value(delta)) * max(1e-6,teacher_data.ExperienceYears[i])^rho)^(1/rho) for i in 1:nrow(teacher_data)
)

# Output results
println("Optimal Parameters:")
println("Base Salary (b): ", value(b))
println("Scale Factor (c): ", value(c))
println("Experience Parameter (alpha): ", value(alpha))
println("Degree Level Parameter (beta): ", value(beta))
println("Effectiveness Weight (delta): ", value(delta))
println("Experience Exponent (p_exp): ", value(p_exp))
println("Degree Level Exponent (p_deg): ", value(p_deg))
println("Total Salary: ", value(optimized_total_salary))
println("Difference: ", value(total_budget) - value(optimized_total_salary))
println("Optimized Total Effectiveness: ", optimized_total_effectiveness)
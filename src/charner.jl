# Burak BASTEM 
# 04.04.2017

using Knet,AutoGrad,ArgParse,Compat

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--datafiles"; nargs='+'; help="If provided, use first file for training, second for dev, others for test.")
        ("--hidden"; nargs='*'; arg_type=Int; default=[128; 128; 128; 128; 128]; help="Sizes of one or more LSTM layers.")        
        ("--epochs"; arg_type=Int; default=600; help="Number of epochs for training.")
        ("--batchsize"; arg_type=Int; default=32; help="Number of sequences to train on in parallel.")
        ("--gclip"; arg_type=Float64; default=1.0; help="Value to clip the gradient norm at.")
        ("--winit"; arg_type=Float64; default=0.1; help="Initial weights set to winit*randn().")
        #("--gcheck"; arg_type=Int; default=0; help="Check N random gradients.")
        ("--dropout"; nargs='*'; arg_type=Float64; default=[0.0; 0.0; 0.0]; help="Dropout probability.")
        ("--seed"; arg_type=Int; default=38; help="Random number seed.")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
    end
    return parse_args(s;as_symbols = true)        
end

function main(args=ARGS)
   opts = parse_commandline()
   println("opts=",[(k,v) for (k,v) in opts]...)
   opts[:seed] > 0 && srand(opts[:seed])
   opts[:atype] = eval(parse(opts[:atype]))    
   # read text and report lengths
   text = map((@compat readstring), opts[:datafiles])
   !isempty(text) && info("Chars read: $(map((f,c)->(basename(f),length(c)),opts[:datafiles],text))")
   # split text into entries
   text = split(text[1], '\n')
   # create character dictionary and report
	char_dic = createCharDictionary(text) 
	info("$(length(char_dic)) unique characters.")	
   # create tag dictionary and report
	tag_dic = createTagDictionary(text)	
	info("$(length(tag_dic)) unique tags.")	
   # create data which consists of (sentence, tagged_sentence) tuples
	data = getData(text, char_dic, tag_dic)
	info("$(length(data)) sentences.")
	# sort data based on length of sentence 
	sort!(data, by=x->length(x[1]))
	# partition data into minibatches
	batches = minibatch(data, char_dic, tag_dic, opts[:batchsize]; atype=opts[:atype])	
	# create model parameters
	model_parameters = createModelParameters(opts[:hidden], length(char_dic), length(tag_dic), opts[:winit]; atype=opts[:atype])
	# create optimizer parameters
	optimizer_parameter = createOptimizerParameters(model_parameters; gclip=opts[:gclip])
	# create initial states and initial cells
	initial_states = createInitialStates(opts[:hidden], opts[:batchsize]; atype=opts[:atype])
	initial_cells = createInitialCells(opts[:hidden], opts[:batchsize]; atype=opts[:atype])
	# train model
	loss, accuracy = evaluate(model_parameters, initial_states, initial_cells, batches; pdrop=opts[:dropout])
	println("epoch=", 0,", loss=", loss, ", accuracy=", accuracy)
	for epoch=1:opts[:epochs]
		shuffle!(batches)
		train(model_parameters, optimizer_parameter, initial_states, initial_cells, batches; pdrop=opts[:dropout])
		loss, accuracy = evaluate(model_parameters, initial_states, initial_cells, batches; pdrop=opts[:dropout])
		println("epoch=", epoch,", loss=", loss, ", accuracy=", accuracy)
	end
end



#####################################################################################################
# creates dictionary for all the characters in entries of given text
#
# julia> createCharDictionary(["Burak 	NNP	I-NP	I-PER", "runs	VBZ	I-VP	O", ".	.	O	O", ""])
# Dict{Char,Int32} with 9 entries:
#  ' ' => 1
#  'B' => 2
#  'u' => 3
#  'r' => 4
#  'a' => 5
#  'k' => 6
#  'n' => 7
#  's' => 8
#  '.' => 9
#
function createCharDictionary(text)
	char_dic = Dict()
	char_dic[' '] = 1 # add whitespace to dictionary
	unknown_char_value = 2
  	for entry in text
  		if(entry != "") # if not end of sentence
  			entry = split(entry, '\t')
  			word = entry[1]
  			for char in word
  				if(unknown_char_value == get!(char_dic, char, unknown_char_value))
	 				unknown_char_value += 1
	 			end
  			end
  		end	
  	end
  	return char_dic    
end

#####################################################################################################
# creates dictionary for all the tags in entries of given text
#
# julia> createTagDictionary(["Burak 	NNP	I-NP	I-PER", "runs	VBZ	I-VP	O", ".	.	O	O", ""])
# Dict{String,Int32} with 2 entries:
#  "PER" => 1
#  "O" => 2
#
function createTagDictionary(text)
	tag_dic = Dict()	
	unknown_tag_value = 1
	for entry in text
  		if(entry != "") # if not end of sentence
  			entry = split(entry, '\t')
  			tag = split(entry[end], '-')[end] # such as B-PER -> PER
  			if(unknown_tag_value == get!(tag_dic, tag, unknown_tag_value))
				unknown_tag_value += 1
			end
  		end	
  	end
	return tag_dic
end

#####################################################################################################
# creates sentences from entries of given text, each character in a sentence replaced with its 
# dictionary number. creates tagged version of corresponding sentence, tag of each character in the 
# sentence replaced with its dictionary number. returns data which contains tuples of sentences and 
# their tagged version. 
#---
# julia> getData(["Burak 	NNP	I-NP	I-PER","runs	VBZ	I-VP	O",".	.	O	O",""], char_dic, tag_dic)
# Any[(Any[2,3,4,5,6,1,4,3,7,8,9],Any[1,1,1,1,1,2,2,2,2,2,2])]
#---
# for clear understanding:
# it returns Any[(sentence, tagged_sentence),...]
#        sentence: "Burak runs." -> Any[2,3,4,5,6,1,4,3,7,8,9]
# tagged_sentence: "PER-PER-PER-PER-PER-O-O-O-O-O-O" -> Any[1,1,1,1,1,2,2,2,2,2,2]

function getData(text, char_dic, tag_dic, whitespace_tag="O")
	#########################
	#original_sent = ""
	#########################
	sentence = []
	tagged_sentence = []
	whitespace_tag = tag_dic[whitespace_tag]
	previous_tag = 0
	data = []
  	for entry in text
  		if(entry == "") # end of sentence
  			push!(data, (sentence[1:end-1], tagged_sentence[1:end-1])) # cut the extra whitespace at the end of sentence
  			#########################
			#println(original_sent)
			#println(sentence)
			#println(tagged_sentence)
			#println()
  			#original_sent = ""
  			#########################
  			sentence = []
  			tagged_sentence = []
			previous_tag = 0
  		else
  			entry = split(entry, '\t')
  			word = entry[1]
  			tag = split(entry[end], '-')[end] # such as "B-PER" -> "PER"
  			tag = tag_dic[tag]		
  			# tag for the space in the phrase
  			if(tag != whitespace_tag && previous_tag == tag)
  				# replace default tag for space with phrase tag
  				tagged_sentence[end] = tag
  			end
  			for char in word
  				push!(sentence, char_dic[char])
  			end
  			push!(sentence, char_dic[' ']) # add the whitespace after word
  			append!(tagged_sentence, fill(tag, length(word)))
  			push!(tagged_sentence, whitespace_tag) # add the whitespace tag after word
  			previous_tag = tag
  			#########################	
  			#original_sent *= word * " "
  			#########################
  		end	
  	end
  	return data
end

#####################################################################################################

function minibatch(data, char_dic, tag_dic, batch_size; atype=KnetArray{Float32})
	char_dic_size = length(char_dic)
	tag_dic_size = length(tag_dic)
	nbatch = div(length(data), batch_size) # ! pay attention, if not divisible
	batches = []
	for i=1:nbatch
		max_length = length(data[i*batch_size][1])
		#########################
		# following line gives this errror:
		#ERROR: LoadError: MethodError: Cannot `convert` an object of type AutoGrad.Rec{Array{Float64,2}} to an object of type BitArray{2}
		#batch = [ falses(batch_size, char_dic_size) for mls=1:max_length ]
		#########################
		batch = Array{Any}(max_length)
		tag_batch = Array{Any}(max_length)
		mask = Array{Any}(max_length)
		for mls=1:max_length
			batch[mls] = convert(atype, falses(batch_size, char_dic_size))
			tag_batch[mls] = convert(atype, falses(batch_size, tag_dic_size))
			mask[mls] = convert(atype, falses(batch_size, 1))			
		end
		for s=1:batch_size
			burak = data[(i-1) * batch_size + s]
			sent = burak[1]
			tagged_sent = burak[2]
			for c=1:length(sent)
				batch[c][s, sent[c]] = 1
				tag_batch[c][s, tagged_sent[c]] = 1
				mask[c] = 1
			end
		end
		push!(batches, (batch, tag_batch, mask))
	end
	return batches
end

#####################################################################################################

function createModelParameters(hidden, char_dic_size, tag_dic_size, winit; atype=KnetArray{Float32})
	parameters = []
	# initialize parameters for BLSTM layers
	for i=1:length(hidden)
		# setting input size
		# input is a one-hot vector for the first BLSTM layer, and
		# concatenation of forward and backward BLSTM outputs of 
		# previous layer.
		input_size = char_dic_size;
		if(i != 1)
			input_size = 2 * hidden[i-1]
		end 
		# parameters of i'th forward blstm layer 
		layer_params = Dict()
		layer_params["f_W_fx"] = convert(atype, randn(input_size, hidden[i]) * winit)
		layer_params["f_W_fh"] = convert(atype, randn(hidden[i], hidden[i]) * winit)
		layer_params["f_w_fc"] = convert(atype, randn(1, hidden[i]) * winit)
		layer_params["f_b_f"] = convert(atype, zeros(1, hidden[i]))
		layer_params["f_W_ix"] = convert(atype, randn(input_size, hidden[i]) * winit)
		layer_params["f_W_ih"] = convert(atype, randn(hidden[i], hidden[i]) * winit)
		layer_params["f_w_ic"] = convert(atype, randn(1, hidden[i]) * winit)
		layer_params["f_b_i"] = convert(atype, zeros(1, hidden[i]))
		layer_params["f_W_cx"] = convert(atype, randn(input_size, hidden[i]) * winit)
		layer_params["f_W_ch"] = convert(atype, randn(hidden[i], hidden[i]) * winit)
		layer_params["f_b_c"] = convert(atype, zeros(1, hidden[i]))
		layer_params["f_W_ox"] = convert(atype, randn(input_size, hidden[i]) * winit)
		layer_params["f_W_oh"] = convert(atype, randn(hidden[i], hidden[i]) * winit)
		layer_params["f_w_oc"] = convert(atype, randn(1, hidden[i]) * winit)
		layer_params["f_b_o"] = convert(atype, zeros(1, hidden[i]))
		# parameters of i'th backward blstm layer
		layer_params["b_W_fx"] = convert(atype, randn(input_size, hidden[i]) * winit)
		layer_params["b_W_fh"] = convert(atype, randn(hidden[i], hidden[i]) * winit)
		layer_params["b_w_fc"] = convert(atype, randn(1, hidden[i]) * winit)
		layer_params["b_b_f"] = convert(atype, zeros(1, hidden[i]))
		layer_params["b_W_ix"] = convert(atype, randn(input_size, hidden[i]) * winit)
		layer_params["b_W_ih"] = convert(atype, randn(hidden[i], hidden[i]) * winit)
		layer_params["b_w_ic"] = convert(atype, randn(1, hidden[i]) * winit)
		layer_params["b_b_i"] = convert(atype, zeros(1, hidden[i]))
		layer_params["b_W_cx"] = convert(atype, randn(input_size, hidden[i]) * winit)
		layer_params["b_W_ch"] = convert(atype, randn(hidden[i], hidden[i]) * winit)
		layer_params["b_b_c"] = convert(atype, zeros(1, hidden[i]))
		layer_params["b_W_ox"] = convert(atype, randn(input_size, hidden[i]) * winit)
		layer_params["b_W_oh"] = convert(atype, randn(hidden[i], hidden[i]) * winit)
		layer_params["b_w_oc"] = convert(atype, randn(1, hidden[i]) * winit)
		layer_params["b_b_o"] = convert(atype, zeros(1, hidden[i]))
		push!(parameters, layer_params)
	end
	# initialize parameters for last layer (for softmax)
	softmax_params = Dict()
	softmax_params["W"] = convert(atype, randn(2 * hidden[end], tag_dic_size) * winit)
	softmax_params["b"] = convert(atype, zeros(1, tag_dic_size))
	push!(parameters, softmax_params)
	prms = parameters[1]
	return parameters
end

#####################################################################################################
# creates initial states (h_t-1 or h_t+1) for BLSTM layers

function createInitialStates(hidden, batch_size; atype=KnetArray{Float32})
	initial_states = []
	for i=1:length(hidden)
		push!(initial_states, convert(atype, zeros(batch_size, hidden[i])))
	end
	return initial_states
end

#####################################################################################################
# creates initial cells (c_t-1 or c_t+1) for BLSTM layers

function	createInitialCells(hidden, batch_size; atype=KnetArray{Float32})
	initial_cells = []
	for i=1:length(hidden)
		push!(initial_cells, convert(atype, zeros(batch_size, hidden[i])))
	end
	return initial_cells
end
#####################################################################################################
# creates parameters for stochastic optimization algorithm (SOA)
# uses Adam Algorithm and default algorithm parameters

function createOptimizerParameters(model_parameters; gclip=0)
	params = []
	for i=1:length(model_parameters)
	layer_params = Dict()
		for key in keys(model_parameters[i])
			layer_params[key] = Adam(; gclip=gclip)
		end
	push!(params, layer_params)
	end
	return params
end

#####################################################################################################
# LSTM with peephole connections
#
# prms: parameters
#			W-> weights such as prms["W_fx"]: input to forget gate weight matrix 
#			w-> weights	such as prms["w_fc"]: cell to forget gate weight vector
#     	b-> biases such as prms["b_f"]: bias to forget gate
# h_t_1: hidden state vector at time t-1 or t+1 depending on forward or backward lstm respectively
# c_t_1: cell vector at time t-1 or t+1 depending on forward or backward lstm respectively
#   x_t: input vector at time t
#   f_t: forget gate at time t
#   i_t: input gate at time t
#   c_t: cell vector at time t
#   o_t: output gate at time t
#   h_t: hidden state vector at time t

function lstm(prms, h_t_1, c_t_1, x_t; backward=false)
	# d: direction, it is forward by default
	d = "f_"
	if(backward)
		d = "b_"
	end
	f_t = sigm(x_t * prms[d*"W_fx"] + h_t_1 * prms[d*"W_fh"] + prms[d*"w_fc"] .* c_t_1 .+ prms[d*"b_f"])
	i_t = sigm(x_t * prms[d*"W_ix"] + h_t_1 * prms[d*"W_ih"] + prms[d*"w_ic"] .* c_t_1 .+ prms[d*"b_i"])
	c_t = f_t .* c_t_1 + i_t .* tanh(x_t * prms[d*"W_cx"] + h_t_1 * prms[d*"W_ch"] .+ prms[d*"b_c"])
	o_t = sigm(x_t * prms[d*"W_ox"] + h_t_1 * prms[d*"W_oh"] + prms[d*"w_oc"] .* c_t .+ prms[d*"b_o"])
	h_t = o_t .* tanh(c_t)
	return (h_t, c_t)
end

#####################################################################################################

function blstm(parameters, initial_hidden, initial_cell, sequence)
	# forward lstm
	forward_hiddens = Array{Any}(length(sequence))
	forward_hiddens[1], forward_cell = lstm(parameters, initial_hidden, initial_cell, sequence[1])
	for i=2:length(sequence)
		forward_hiddens[i], forward_cell = lstm(parameters, forward_hiddens[i-1], forward_cell, sequence[i])
	end
	# backward lstm
	backward_hiddens = Array{Any}(length(sequence))
	backward_hiddens[end], backward_cell = lstm(parameters, initial_hidden, initial_cell, sequence[end]; backward=true)
	for i=length(sequence)-1:-1:1
		backward_hiddens[i], backward_cell = lstm(parameters, backward_hiddens[i+1], backward_cell, sequence[i]; backward=true)
	end
	return forward_hiddens, backward_hiddens
end

#####################################################################################################

function predict(parameters, initial_hiddens, initial_cells, sequence, mask; pdrop=[0.0 0.0 0.0])
	seq = copy(sequence)
	for i=1:length(parameters)-1
		map!(x -> dropout(x, pdrop[1]), seq)
		forward_hiddens, backward_hiddens = blstm(parameters[i], initial_hiddens[i], initial_cells[i], seq)
		for j=1:length(seq)
			seq[j] = hcat(dropout(forward_hiddens[j], pdrop[2]), dropout(backward_hiddens[j], pdrop[3])) .* mask[j]
		end
	end
	last_layer = parameters[end]
	for j=1:length(sequence)
			seq[j] = seq[j] * last_layer["W"] .+ last_layer["b"]
	end
	return seq
end

#####################################################################################################

function loss(parameters, initial_hiddens, initial_cells, sequence, ygold, mask; pdrop=[0.0 0.0 0.0])
    total = 0.0
    count = 0
    ypred = predict(parameters, initial_hiddens, initial_cells, sequence, mask; pdrop=pdrop)
    for i=1:length(sequence)
        ynorm = logp(ypred[i], 2) # ypred .- log(sum(exp(ypred),2))
        total += sum(ygold[i] .* ynorm)
        count += sum(ygold[i])
    end
    return -total / count
end

#####################################################################################################

lossGradient = grad(loss)

#####################################################################################################

function train(model_params, optimizer_params, initial_hiddens, initial_cells, batches; pdrop=[0.0 0.0 0.0])
	for minibatch in batches	
		grad_loss = lossGradient(model_params, initial_hiddens, initial_cells, minibatch[1], minibatch[2], minibatch[3]; pdrop=pdrop)		
		# @show gradcheck(loss, model_params, initial_cells, initial_hiddens, minibatch[1], minibatch[2], minibatch[3]; verbose=true, atol=0.01)
		for i=1:length(model_params)
			layer_model = model_params[i]
			layer_grad_loss = grad_loss[i]
			layer_optimizer = optimizer_params[i]
			for k in keys(layer_model)
				update!(layer_model[k], layer_grad_loss[k], layer_optimizer[k])
			end
		end
		
	end
end

#####################################################################################################
function evaluate(model_params, initial_hiddens, initial_cells, batches; pdrop=[0.0 0.0 0.0])
	total_loss = 0.0
	total_correct = 0
	total_count = 0
	for minibatch in batches
		x = minibatch[1]
		ygold = minibatch[2]
		mask = minibatch[3]	
		minibatch_total_loss = 0.0
		minibatch_total_correct = 0
		minibatch_count = 0
		ypred = predict(model_params, initial_hiddens, initial_cells, x, mask; pdrop=pdrop)
		for i=1:length(x)
			ynorm = logp(ypred[i], 2) # ypred .- log(sum(exp(ypred),2))
			minibatch_total_loss += sum(ygold[i] .* ynorm)			
			minibatch_total_correct += sum(ygold[i] .* (ypred[i] .== maximum(ypred[i],2)))
			minibatch_count += sum(ygold[i])		
		end
		total_loss += -minibatch_total_loss / minibatch_count
		total_correct += minibatch_total_correct
		total_count += minibatch_count
	end
	softmax_loss = total_loss/length(batches)
	accuracy = total_correct/total_count
	return softmax_loss, accuracy
end
#####################################################################################################

main()

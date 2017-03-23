using Knet,AutoGrad,ArgParse,Compat

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--datafiles"; nargs='+'; help="If provided, use first file for training, second for dev, others for test.")
        ("--generate"; arg_type=Int; default=500; help="If non-zero generate given number of characters.")
        ("--hidden";  arg_type=Int; default=256; help="Sizes of one or more LSTM layers.")
        ("--epochs"; arg_type=Int; default=3; help="Number of epochs for training.")
        ("--batchsize"; arg_type=Int; default=32; help="Number of sequences to train on in parallel.")
        ("--seqlength"; arg_type=Int; default=25; help="Number of steps to unroll the network for.")
        ("--decay"; arg_type=Float64; default=0.9; help="Learning rate decay.")
        ("--lr"; arg_type=Float64; default=1e-1; help="Initial learning rate.")
        ("--gclip"; arg_type=Float64; default=3.0; help="Value to clip the gradient norm at.")
        ("--winit"; arg_type=Float64; default=0.1; help="Initial weights set to winit*randn().")
        ("--gcheck"; arg_type=Int; default=0; help="Check N random gradients.")
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
   
	text = split(text[1], '\n') # split into sentences
	sort!(text, by=actualLength)
	vocab = createVocabulary(text)
	info("$(length(vocab)) unique chars.")
	tags = createTagDictionary(text)
	info("$(length(tags)) unique tags.")
	data = getData(text, vocab, tags)
	data = minibatch(data, vocab, tags, opts[:batchsize]) 
end

###################################################################################################

# This function is not used. It is replaced.
# This function creates tag dictionary, removes tags from data and creates tagged version of
# data. Since it does all the things together and is hard to understand, I seperated all the
# functionalty.
#function getData_old(text)
#	text = split(text[1], '\n') # split into sentences
#	data = Any[]
#	tag_dic = Dict{String,Int}()
#	tag_dic["O"] = 0
#	tag_value = 1
#	for sentence in text
#		pure_sentence = sentence
#		for regex_match in eachmatch(r"\[\S\S\S (.*?) \]", sentence)
#			pure_sentence = replace(pure_sentence, regex_match.match, regex_match.captures[1])
#		end
#		pure_sentence = join(split(pure_sentence), " ")
#		tagged_sentence = zeros(Int,length(pure_sentence))
#		for regex_match in eachmatch(r"\[\S\S\S (.*?) \]", sentence)
#			tag = regex_match.match[2:4]
#			(tag_value == get!(tag_dic, tag, tag_value) ? tag_value = tag_value + 1 : 0) # <- else 0 search for it
#			#range = search(pure_sentence, regex_match.captures[1]) <-- does not work! UTF8
#			# start: search does not give correct indices for Turkish characters -> a workaround
#			capture = join(split(regex_match.captures[1]), " ") # there are tags contain multiple whitespaces			
#			pure_sentence_uint = transcode(UInt32, pure_sentence)
#			capture_uint = transcode(UInt32, capture)
#			for i=1:length(pure_sentence_uint)-length(capture_uint)+1
#				range = i:i+length(capture_uint)-1
#				if(pure_sentence_uint[range] == capture_uint)
#					tagged_sentence[range] = tag_dic[tag]
#				end
#			end
#		end
#		# end: search does not give correct indices for Turkish characters -> a workaround
#		push!(data, (pure_sentence, tagged_sentence))
#	end
#	return data
#end

###################################################################################################

function clearTags(sentence)
	pure_sentence = sentence
	for regex_match in eachmatch(r"\[\S\S\S (.*?) \]", sentence)
		pure_sentence = replace(pure_sentence, regex_match.match, regex_match.captures[1])
	end
	pure_sentence = join(split(pure_sentence), " ")
	return pure_sentence
end

###################################################################################################

function actualLength(sentence)
	return length(clearTags(sentence))
end

###################################################################################################

function createVocabulary(text)
    vocab = Dict{Char,Int}()
    char_value = 1
	 for sentence in text
	 	pure_sentence = clearTags(sentence)
	 	for character in pure_sentence
	 		if(char_value == get!(vocab, character, char_value))
	 			char_value = char_value + 1
	 		end
	 	end
	 end
    return vocab
end

###################################################################################################

function createTagDictionary(text)
	tags = Dict{String,Int}()
	tags["O"] = 1
	tag_value = 2
	for sentence in text
		for regex_match in eachmatch(r"\[\S\S\S (.*?) \]", sentence)
			tag = regex_match.match[2:4]
			if(tag_value == get!(tags, tag, tag_value))
				tag_value = tag_value + 1
			end
		end
	end
	return tags
end

###################################################################################################

function getData(text, vocab, tags)
	data = Any[]
	for sentence in text
		pure_sentence = clearTags(sentence)
		tagged_sentence = ones(Int,length(pure_sentence))
		for regex_match in eachmatch(r"\[\S\S\S (.*?) \]", sentence)
			named_entity = join(split(regex_match.captures[1]), " ") # there are tags contain multiple whitespaces
			#range = search(pure_sentence, regex_match.captures[1]) <-- does not work! UTF8
			# search does not give correct indices for Turkish characters -> a workaround solution
			pure_sent_uint = transcode(UInt32, pure_sentence)
			ne_uint = transcode(UInt32, named_entity)
			for i=1:length(pure_sent_uint)-length(ne_uint)+1
				range = i:i+length(ne_uint)-1
				if(pure_sent_uint[range] == ne_uint)
					tagged_sentence[range] = tags[regex_match.match[2:4]]
				end
			end
		end	
		push!(data, (pure_sentence, tagged_sentence))	
	end
	return data
end

###################################################################################################

function minibatch(data, vocab, tags, batch_size)
	batched_data = Any[]	
	for i=1:batch_size:length(data)
		minibatch = Any[]
		for (sentence, tagged_sentence) in data[i:(i+batch_size-1 < length(data) ? i+batch_size-1 : length(data))]
			x = Any[]
			y = Any[]
			for character in sentence
				char_vector = zeros(Int,length(vocab))
				char_vector[vocab[character]] = 1
				push!(x, char_vector)
			end
			for tag in tagged_sentence
				one_hot = zeros(Int,length(tags))
				one_hot[tag] = 1
				push!(y, one_hot)
			end
			push!(minibatch, (x, y))		
		end
		push!(batched_data, minibatch)
	end
	return batched_data
end

###################################################################################################

main()

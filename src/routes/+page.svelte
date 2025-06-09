<script lang="ts">
	import { DataTable, type ColumnDef } from '@careswitch/svelte-data-table';

	interface TableRow {
		id: number;
		name: string;
		duration: number;
		avg_uncertainty: number | null | undefined;
		max_uncertainty: number | null | undefined;
		gps: string | null | undefined;
		raw_sequence: any;
	}

	let isLoading = $state(true);
	let error = $state<string | null>(null);

	let isCompactView = $state(true);

	let modalElement: HTMLElement | undefined = $state();
	let modalImageUrl: string | null = $state(null);

	$effect(() => {
		if (modalImageUrl && modalElement) {
			modalElement.focus();
		}
	});

	const emptyVal = ' ';
	const nullToEmpty = (val: any) => {
		if (val === undefined || val === null) return emptyVal;
		return val;
	};

	const emptyAwareSorter = (a: number | string, b: number | string, rowA: any, rowB: any) => {
		const aIsEmpty = a === emptyVal;
		const bIsEmpty = b === emptyVal;

		if (aIsEmpty && bIsEmpty) return 0;

		if (aIsEmpty || bIsEmpty) {
			// table library switches rows for asc and desc. TODO: need to check if this works.
			const isAscendingSort = rowA.id > rowB.id;
			const direction = isAscendingSort ? 1 : -1;
			// In ascending (direction=1), empty values should go last (return 1 for aIsEmpty).
			// In descending (direction=-1), empty values should go first (return -1 for aIsEmpty).
			return aIsEmpty ? 1 * direction : -1 * direction;
		}

		return Number(a) - Number(b);
	};

	const containsFilter = (value: string, filterValue: string, row: any) => {
		if (filterValue === '') return true;
		return value.includes(filterValue);
	};

	const columnsConfig: ColumnDef<TableRow>[] = [
		{
			id: 'id',
			key: 'id',
			name: 'FMC ID',
			sortable: true,
			getValue: (r: TableRow) => String(r.id),
			filter: containsFilter
		},
		{ id: 'name', key: 'name', name: 'Sequence Name', sortable: true, filter: containsFilter },
		{
			id: 'duration',
			key: 'duration',
			name: 'Duration (sec)',
			sortable: true,
			sorter: emptyAwareSorter,
			getValue: (r: TableRow) => String(r.duration),
			filter: containsFilter
		},
		{
			id: 'avg_uncertainty',
			key: 'avg_uncertainty',
			name: 'Average Uncertainty',
			sortable: true,
			sorter: emptyAwareSorter,
			getValue: (r: TableRow) => String(nullToEmpty(r.avg_uncertainty)),
			filter: containsFilter
		},
		{
			id: 'max_uncertainty',
			key: 'max_uncertainty',
			name: 'Max Uncertainty',
			sortable: true,

			sorter: emptyAwareSorter,
			getValue: (r: TableRow) => String(nullToEmpty(r.max_uncertainty)),
			filter: containsFilter
		}
	];

	let table = $state(
		new DataTable({
			data: [] as TableRow[],
			columns: columnsConfig
		})
	);

	$effect(() => {
		(async () => {
			isLoading = true;
			error = null;

			try {
				const resp = await fetch('http://157.90.124.55:8000/sequences/');
				if (!resp.ok) {
					throw new Error(`Failed to fetch data. Status: ${resp.status}`);
				}

				let fetchedData: any[] = await resp.json();

				const formattedData: TableRow[] = fetchedData.map((seq) => ({
					id: seq.id,
					name: seq.name,
					duration: seq.duration,
					avg_uncertainty: seq.latest_uncertainty?.avg_uncertainty ?? null,
					max_uncertainty: seq.latest_uncertainty?.max_uncertainty ?? null,
					gps: seq.gps,
					raw_sequence: seq
				}));

				table = new DataTable({
					data: formattedData,
					columns: columnsConfig
				});
			} catch (e: any) {
				error = e.message || 'An unknown error occurred.';
			} finally {
				isLoading = false;
			}
		})();
	});

	const visibleRowIds = $derived(table.rows.map((row) => row.id));

	let selectedIds = $state(new Set<number>());

	const isAllSelected = $derived(
		visibleRowIds.length > 0 && visibleRowIds.every((id) => selectedIds.has(id))
	);

	const isSomeSelected = $derived(
		visibleRowIds.length > 0 && visibleRowIds.some((id) => selectedIds.has(id))
	);

	function handleSelectAll() {
		if (isAllSelected) {
			// Select all is toggle. If all are selected, deselect all visible rows
			const newSelectedIds = new Set(selectedIds);
			for (const id of visibleRowIds) {
				newSelectedIds.delete(id);
			}
			selectedIds = newSelectedIds;
		} else {
			const newSelectedIds = new Set(selectedIds);
			for (const id of visibleRowIds) {
				newSelectedIds.add(id);
			}
			selectedIds = newSelectedIds;
		}
	}

	function toggleRowSelection(id: number) {
		const newSelectedIds = new Set(selectedIds);
		if (newSelectedIds.has(id)) {
			newSelectedIds.delete(id);
		} else {
			newSelectedIds.add(id);
		}
		selectedIds = newSelectedIds;
	}

	function getSortIcon(columnKey: string) {
		const sortState = table.getSortState(columnKey);
		if (sortState === 'asc') return '↑';
		if (sortState === 'desc') return '↓';
		return '';
	}
</script>

{#if isLoading}
	<div class="flex items-center justify-center p-10">
		<div class="text-lg font-semibold text-gray-500">Loading data...</div>
	</div>
{:else if error}
	<div class="rounded-md border border-red-400 bg-red-50 p-4 text-red-700">
		<h3 class="font-bold">An Error Occurred</h3>
		<p>{error}</p>
	</div>
{:else}
	<div class="sticky top-2 flex min-w-screen flex-row space-x-2">
		<button
			class="mb-2 h-12 w-64 rounded bg-blue-200 p-2"
			onclick={() => (isCompactView = !isCompactView)}
		>
			{#if isCompactView}
				Switch to Normal View
			{:else}
				Switch to Compact View
			{/if}
		</button>
		<div class="mb-4 w-full">
			<label for="search" class="mb-2 block text-sm font-medium text-gray-700">Filter:</label>
			<input
				id="search"
				type="text"
				bind:value={table.globalFilter}
				placeholder="e.g., ZEUS"
				class="w-1/3 rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:ring-2 focus:ring-blue-500 focus:outline-none"
			/>
		</div>
	</div>

	<div class="overflow-x-auto">
		<table class="w-full border-collapse">
			<thead>
				<tr>
					<th class="border border-gray-300 bg-gray-100 px-2 py-2 text-left">
						<input
							type="checkbox"
							aria-label="Select all"
							checked={isAllSelected}
							indeterminate={isSomeSelected && !isAllSelected}
							onchange={handleSelectAll}
							class="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
						/>
					</th>
					{#each table.columns as column (column.id)}
						<th class="border border-gray-300 bg-gray-100 px-2 py-2 text-left whitespace-nowrap">
							{#if column.sortable}
								<button
									type="button"
									class="flex items-center gap-1 font-medium hover:text-blue-600 focus:text-blue-600 focus:outline-none"
									onclick={() => table.toggleSort(column.key)}
									aria-label="Sort by {column.name}"
								>
									{column.name}
									<span class="text-sm text-gray-400" aria-hidden="true">
										{getSortIcon(column.key)}
									</span>
								</button>
							{:else}
								{column.name}
							{/if}
						</th>
					{/each}
					<th class="border border-gray-300 bg-gray-100 px-2 py-2 text-left font-medium">Map</th>
				</tr>
				<tr class="bg-gray-50">
					<!-- Empty cell for the checkbox column -->
					<th class="border border-gray-300 p-1"></th>
					{#each table.columns as column (column.id)}
						<th class="border border-gray-300 p-1">
							<input
								type="text"
								placeholder="Filter..."
								class="w-full rounded border-gray-200 px-2 py-1 text-sm font-normal shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
								onchange={(e) => table.setFilter(column.key, [e.currentTarget.value])}
								oninput={(e) => table.setFilter(column.key, [e.currentTarget.value])}
							/>
						</th>
					{/each}
					<th class="border border-gray-300 p-1"></th>
				</tr>
			</thead>
			<tbody>
				{#each table.rows as row (row.id)}
					<tr class="hover:bg-gray-50">
						<td class="w-10 border border-gray-300 px-2 py-2">
							<input
								type="checkbox"
								aria-label="Select row {row.id}"
								checked={selectedIds.has(row.id)}
								onchange={() => toggleRowSelection(row.id)}
								class="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
							/>
						</td>
						{#each table.columns as column (column.id)}
							<td class="border border-gray-300 px-2 py-2 whitespace-nowrap">
								{row[column.key]}
							</td>
						{/each}
						<td class="border border-gray-300 px-1 py-1">
							{#if row.gps}
								<button
									type="button"
									class="transition hover:opacity-80"
									onclick={() =>
										(modalImageUrl = `http://157.90.124.55:8000/sequence_maps/${row.id}`)}
									aria-label="View larger map for {row.name}"
								>
									<img
										class:w-[50px]={isCompactView}
										class:w-[500px]={!isCompactView}
										src={`http://157.90.124.55:8000/sequence_maps/${row.id}`}
										alt="Driving map for {row.id}"
									/>
								</button>
							{/if}
						</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</div>

	<div class="mt-4">
		<p class="text-sm text-gray-700">
			<span class="font-semibold">Selected IDs:</span>
			{Array.from(selectedIds).join(', ') || 'None'}
		</p>
	</div>
	<div class="mt-4">
		<p class="text-sm text-gray-700">
			<span class="font-semibold">Scene Count:</span>
			{table.allRows.length}
		</p>
	</div>
{/if}

{#if modalImageUrl}
	<div
		bind:this={modalElement}
		class="bg-opacity-75 fixed inset-0 z-50 flex items-center justify-center bg-black p-4"
		onclick={() => (modalImageUrl = null)}
		onkeydown={(e) => e.key === 'Escape' && (modalImageUrl = null)}
		role="dialog"
		aria-modal="true"
		tabindex="0"
		aria-label="Close image viewer (click or press Escape)"
	>
		<div
			class="relative"
			onclick={(e) => e.stopPropagation()}
			onkeydown={(e) => e.stopPropagation()}
			role="presentation"
		>
			<img
				src={modalImageUrl}
				alt="Full-size map"
				class="block h-[90vh] max-w-[90vw] rounded-lg shadow-2xl"
			/>
			<button
				type="button"
				class="absolute -top-3 -right-3 flex h-8 w-8 items-center justify-center rounded-full bg-white text-gray-700 shadow-lg hover:bg-gray-200 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:outline-none"
				onclick={() => (modalImageUrl = null)}
				aria-label="Close image viewer"
			>
				<span aria-hidden="true">×</span>
				<span class="sr-only">Close</span>
			</button>
		</div>
	</div>
{/if}

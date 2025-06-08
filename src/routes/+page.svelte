<script lang="ts">
	import { DataTable } from '@careswitch/svelte-data-table';

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

	const table = $state(
		new DataTable({
			data: [
				{ id: 1, name: 'John Doe___', duration: 3, avg_uncertainty: undefined, max_uncertainty: 2 },
				{ id: 2, name: 'Jane Doe___', duration: 3, avg_uncertainty: 2, max_uncertainty: undefined },
				{ id: 3, name: 'Peter Jones', duration: 5, avg_uncertainty: undefined, max_uncertainty: 2 },
				{ id: 4, name: 'Susan Smith', duration: 4, avg_uncertainty: undefined, max_uncertainty: 2 },
				{ id: 5, name: 'David Millr', duration: 3, avg_uncertainty: undefined, max_uncertainty: 2 },
				{ id: 6, name: 'Mary Brown_', duration: 7, avg_uncertainty: 1, max_uncertainty: 2 }
			],
			columns: [
				{
					id: 'id',
					key: 'id',
					name: 'FMC ID',
					sortable: true,
					getValue: (r) => String(r.id),
					filter: containsFilter
				},
				{ id: 'name', key: 'name', name: 'Sequence Name', sortable: true, filter: containsFilter },
				{
					id: 'duration',
					key: 'duration',
					name: 'Duration (sec)',
					sortable: true,
					sorter: emptyAwareSorter,
					getValue: (r) => String(r.duration),
					filter: containsFilter
				},
				{
					id: 'avg_uncertainty',
					key: 'avg_uncertainty',
					name: 'Average Uncertainty',
					sortable: true,
					sorter: emptyAwareSorter,
					getValue: (row) => String(nullToEmpty(row.avg_uncertainty)),
					filter: containsFilter
				},
				{
					id: 'max_uncertainty',
					key: 'max_uncertainty',
					name: 'Max Uncertainty',
					sortable: true,

					sorter: emptyAwareSorter,
					getValue: (row) => String(nullToEmpty(row.max_uncertainty)),
					filter: containsFilter
				}
			]
		})
	);

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

<div class="mb-4">
	<label for="search" class="mb-2 block text-sm font-medium text-gray-700">Filter:</label>
	<input
		id="search"
		type="text"
		bind:value={table.globalFilter}
		placeholder="e.g., ZEUS"
		class="w-1/3 rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:ring-2 focus:ring-blue-500 focus:outline-none"
	/>
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
			</tr>
			<tr class="bg-gray-50">
				<th class="border border-gray-300 p-1"></th>
				<!-- Empty cell for the checkbox column -->
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
